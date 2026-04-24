"""Thread-keyed QuickJS REPL registry, console bridge, and result formatter.

Kept separate from ``middleware.py`` so the REPL mechanics stay testable
without constructing an agent or wiring up LangGraph state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from quickjs_rs import (
    UNDEFINED,
    ConcurrentEvalError,
    Context,
    DeadlockError,
    HostCancellationError,
    HostError,
    JSError,
    MarshalError,
    MemoryLimitError,
    ModuleScope,
    Runtime,
    ThreadWorker,
)
from quickjs_rs import (
    TimeoutError as QJSTimeoutError,
)

from langchain_quickjs._ptc import to_camel_case
from langchain_quickjs._skills import LoadedSkill, SkillLoadError, aload_skill

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.skills import SkillMetadata
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

# Sentinel returned by the formatter when the underlying value was a
# function/circular ref that couldn't be auto-marshaled. We format it as
# a handle-shaped result so the model sees "you got back a function" rather
# than nothing.
_HANDLE_PLACEHOLDER = "[unmarshalable value]"

_TRUNCATE_MARKER = "… [truncated {n} chars]"


@dataclass
class EvalOutcome:
    """Normalized result of a single REPL eval.

    Exactly one of ``result`` / ``error`` is meaningful per call; ``stdout``
    is collected from ``console.*`` regardless.
    """

    stdout: str = ""
    result: str | None = None
    result_kind: str | None = None  # "handle" when marshaling fell back
    error_type: str | None = None
    error_message: str = ""
    error_stack: str | None = None


class _ConsoleBuffer:
    """Accumulates ``console.*`` output between evals.

    Shared by the three host functions we install on each context. We don't
    bother distinguishing log/warn/error in the output format — the model
    does not care about the level, and flattening keeps the returned string
    smaller.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []

    def append(self, level: str, args: tuple[Any, ...]) -> None:
        del level  # flattened; see class docstring
        self._lines.append(" ".join(_stringify(a) for a in args))

    def drain(self) -> str:
        if not self._lines:
            return ""
        out = "\n".join(self._lines)
        self._lines.clear()
        return out


def _format_handle(handle: Any) -> str:
    """Describe a ``Handle`` value in REPL-style shorthand.

    Caller owns the handle's lifetime; we only read from it.
    """
    kind = handle.type_of
    if kind == "function":
        # Arity is convenient context when the model wants to call the
        # thing back. Fall back gracefully if .length is absent.
        try:
            arity_h = handle.get("length")
            try:
                arity = arity_h.to_python()
            finally:
                arity_h.dispose()
            return f"[Function] arity={arity}"
        except Exception:  # noqa: BLE001 — best-effort
            return "[Function]"
    return f"[{kind}]"


def _stringify(value: Any) -> str:
    """Best-effort string form for a console arg or eval result.

    QuickJS auto-marshals primitives and plain objects through msgpack, so
    everything we see here is already a Python value. Formatting choices
    match Node's REPL rather than Python's ``repr``:

    - ``None`` → ``"null"`` (the model expects JS-shaped output)
    - ``UNDEFINED`` → ``"undefined"``
    - Booleans → ``"true"`` / ``"false"``
    - Whole-valued floats → integer form (``42.0`` → ``"42"``). JS has no
      integer type, so every ``1 + 1`` comes back as a float; without this
      the model sees ``42.0`` where a human would expect ``42``. Applied
      recursively inside lists and dicts.
    """
    return _format_jsvalue(value)


def _format_jsvalue(value: Any) -> str:
    if value is None:
        return "null"
    if value is UNDEFINED:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Preserve ±inf / NaN: .is_integer() returns False for NaN and inf,
        # so they fall through to str().
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        # Top-level strings render bare (matches what a REPL user expects
        # when they eval a string expression); nested strings get quoted
        # so ``[1, "a"]`` renders as ``[1, "a"]`` not ``[1, a]``.
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_format_nested(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{" + ", ".join(f"{k}: {_format_nested(v)}" for k, v in value.items()) + "}"
        )
    return repr(value)


def _format_nested(value: Any) -> str:
    """Like ``_format_jsvalue`` but quotes nested strings."""
    if isinstance(value, str):
        return f'"{value}"'
    return _format_jsvalue(value)


def _normalize_tool_input(raw: Any) -> dict[str, Any]:
    """Coerce whatever JS passed into ``tools.X(...)`` to a dict.

    LangChain tools accept a dict. QuickJS marshals JS objects to dicts
    already; we just want to guard against the model passing ``null``,
    ``undefined``, a bare string, or a number (none of which a well-
    formed tool call should produce, but the model is the model).
    """
    if raw is None or raw is UNDEFINED:
        return {}
    if isinstance(raw, dict):
        return raw
    # Bare scalar / list — wrap under a conventional key so the tool's
    # schema validation produces an informative error rather than a
    # silent miss.
    return {"input": raw}


def _coerce_tool_output(value: Any) -> str:
    """Tools return arbitrary Python; JS-side users expect a string.

    Handles three shapes:

    - ``str`` — pass through unchanged.
    - ``langgraph.types.Command`` — the shape ``task`` / subagent tools
      return. Extract the last ``ToolMessage`` content from
      ``command.update["messages"]`` since that's what the parent agent
      would normally see; the state update itself is intentionally
      dropped — PTC calls happen inside a JS ``await`` and we have
      nowhere to funnel a state mutation back into the parent graph.
    - everything else — ``json.dumps`` for faithful JSON → JS parseable
      round-tripping, falling back to ``str`` on non-serialisable
      values.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Command):
        update = value.update
        if isinstance(update, dict):
            messages = update.get("messages")
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if isinstance(content, str):
                    return content
        # No extractable message — stringify the update for debuggability.
        return str(update)
    # When we invoke with a ToolCall-shaped input, BaseTool wraps the
    # return value in a ToolMessage. Unwrap its content so the JS side
    # sees the raw tool output, not a Python repr of the envelope.
    if isinstance(value, ToolMessage):
        content = value.content
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, default=str)
        except (TypeError, ValueError):
            return str(content)
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _synth_tool_call_id(tool_name: str) -> str:
    """Mint a synthetic tool_call_id for a PTC-driven tool invocation.

    Tools like ``task`` require a non-empty ``tool_call_id`` to stamp
    into their emitted ``ToolMessage``. The real call_id lives on the
    outer ``eval`` tool call; we synthesise a child id so downstream
    state (checkpointer, tracing) can correlate the PTC sub-call back
    to the REPL cell that issued it.
    """
    import uuid

    return f"ptc_{tool_name}_{uuid.uuid4().hex[:8]}"


def _inject_tool_args_for_ptc(
    tool: Any,
    payload: dict[str, Any],
    outer_runtime: Any,
    tool_call_id: str,
) -> dict[str, Any]:
    """Mirror LangGraph's ``ToolNode._inject_tool_args`` for PTC calls.

    LangChain tools that declare ``ToolRuntime`` / ``InjectedState`` /
    ``InjectedStore`` only see those values when a real ``ToolNode``
    wires them in. PTC calls bypass the ToolNode, so we replicate the
    detection logic here. The outer runtime (captured from the active
    ``eval`` tool invocation) provides state/store/context/config;
    ``tool_call_id`` is freshly minted per sub-call.
    """
    try:
        from langgraph.prebuilt.tool_node import _get_all_injected_args
    except ImportError:  # pragma: no cover — langgraph always present
        return payload

    injected = _get_all_injected_args(tool)
    if not injected or outer_runtime is None:
        return payload

    # Build a ToolRuntime matching the outer one but with a fresh
    # tool_call_id. ``type(outer_runtime)`` rather than a literal import
    # so the shape stays in lockstep with whatever langgraph ships.
    derived = type(outer_runtime)(
        state=outer_runtime.state,
        tool_call_id=tool_call_id,
        config=outer_runtime.config,
        context=outer_runtime.context,
        store=outer_runtime.store,
        stream_writer=outer_runtime.stream_writer,
        execution_info=getattr(outer_runtime, "execution_info", None),
        server_info=getattr(outer_runtime, "server_info", None),
    )

    enriched = dict(payload)
    if injected.runtime:
        enriched[injected.runtime] = derived
    # InjectedState: state can be injected under one or more arg names.
    if injected.state:
        for arg_name, state_field in injected.state.items():
            if state_field:
                enriched[arg_name] = (
                    outer_runtime.state.get(state_field)
                    if isinstance(outer_runtime.state, dict)
                    else getattr(outer_runtime.state, state_field, None)
                )
            else:
                enriched[arg_name] = outer_runtime.state
    if injected.store and outer_runtime.store is not None:
        enriched[injected.store] = outer_runtime.store
    return enriched


class _ThreadREPL:
    """One QuickJS context + console buffer, per LangGraph thread.

    All ``ctx.*`` operations are marshalled onto the worker's dedicated
    thread because ``quickjs_rs`` objects are ``!Send``. The public
    methods are safe to call from any thread/loop.
    """

    def __init__(
        self,
        worker: ThreadWorker,
        runtime: Runtime,
        *,
        timeout: float,
        capture_console: bool,
    ) -> None:
        self._worker = worker
        self._runtime = runtime
        # The Context-level ``timeout`` is used as the cumulative budget
        # for sync evals. Async evals pass ``timeout=`` per call so each
        # call gets a fresh budget — matches what a REPL user expects,
        # and what we describe in the system prompt.
        self._per_call_timeout = timeout
        self._capture_console = capture_console
        self._console = _ConsoleBuffer()
        self._ctx: Context | None = None
        # PTC state. ``_registered_tools`` tracks which camel-case names
        # have already had their host-function bridge installed on the
        # QuickJS context. Host functions cannot be un-registered, so we
        # never remove entries from here — changes to the exposed set
        # are reflected by rewriting ``globalThis.tools`` (see
        # install_tools) to include only the currently-active subset.
        self._registered_tools: dict[str, BaseTool] = {}
        self._active_tool_names: frozenset[str] = frozenset()
        # Tracks whether ``globalThis.tools`` has been assigned at least
        # once. Distinct from ``_active_tool_names`` so the first call
        # with an empty tool set still installs ``tools = {}`` (otherwise
        # ``typeof tools.X`` throws ReferenceError instead of returning
        # ``"undefined"``).
        self._tools_installed: bool = False
        # Outer ToolRuntime captured for the current eval. PTC bridges
        # forward it into their tool calls so `task`/subagent tools see
        # graph state, store, context, etc. Set via ``set_outer_runtime``
        # from the middleware's tool handler immediately before eval.
        self._outer_runtime: ToolRuntime | None = None
        # Context creation + console install must happen on the worker
        # thread. Block caller here so the REPL is ready to use when
        # __init__ returns.
        worker.run_sync(self._ainit())

    async def _ainit(self) -> None:
        self._ctx = self._runtime.new_context(timeout=self._per_call_timeout)
        if self._capture_console:
            self._install_console()

    def _install_console(self) -> None:
        ctx = self._ctx
        buf = self._console

        @ctx.function(name="__console_log")
        def _log(*args: Any) -> None:
            buf.append("log", args)

        @ctx.function(name="__console_warn")
        def _warn(*args: Any) -> None:
            buf.append("warn", args)

        @ctx.function(name="__console_error")
        def _error(*args: Any) -> None:
            buf.append("error", args)

        # Install the JS-level console object. We do this via a separate
        # eval because register_host_function only puts the callable on the
        # global object under its given name; ``globalThis.console`` needs
        # to exist as a normal object for idiomatic JS. Trailing primitive
        # keeps the eval's result marshalable — assigning an object would
        # bubble a MarshalError we'd have to special-case.
        ctx.eval(
            "globalThis.console = {"
            " log: __console_log,"
            " warn: __console_warn,"
            " error: __console_error,"
            "}; undefined"
        )

    def install_tools(self, tools: Sequence[BaseTool]) -> None:
        """Expose ``tools`` as ``globalThis.tools.<camelCase>`` in the REPL.

        Idempotent per (camelName, tool identity). Safe to call on every
        model-call turn; we diff against the current active set and only
        (a) register new host-function bridges for tools we haven't seen
        before and (b) rewrite ``globalThis.tools`` when the active-name
        set changes. Hot path cost when nothing changes: one frozenset
        equality check.
        """
        self._worker.run_sync(self._ainstall_tools(tools))

    async def _ainstall_tools(self, tools: Sequence[BaseTool]) -> None:
        ctx = self._ctx
        name_to_tool: dict[str, BaseTool] = {to_camel_case(t.name): t for t in tools}
        target_names = frozenset(name_to_tool)
        if target_names == self._active_tool_names and self._tools_installed:
            # Fast path: stable toolset, nothing to do. Keep the bridge's
            # dispatch target pointer current in case tool objects rotate
            # while keeping the same names. Guard with ``_tools_installed``
            # so the empty → empty transition on first call still installs
            # a ``tools = {}`` global — otherwise ``typeof tools.x`` hits a
            # ReferenceError instead of returning "undefined".
            self._registered_tools.update(name_to_tool)
            return

        # Register host-function bridges for tools we haven't seen before.
        for camel, tool in name_to_tool.items():
            if camel not in self._registered_tools:
                self._register_tool_bridge(camel)
            self._registered_tools[camel] = tool

        # Rewrite globalThis.tools. Building the object inside a single
        # eval keeps assignments atomic from the model's point of view —
        # there's no moment where tools is half-populated. The trailing
        # ``undefined`` sidesteps the MarshalError on object returns
        # (same trick as the console install).
        if target_names:
            pairs = ", ".join(f"{camel}: __tools_{camel}" for camel in target_names)
            ctx.eval(f"globalThis.tools = {{ {pairs} }}; undefined")
        else:
            ctx.eval("globalThis.tools = {}; undefined")
        self._active_tool_names = target_names
        self._tools_installed = True

    def set_outer_runtime(self, runtime: ToolRuntime | None) -> None:
        """Record the outer ``ToolRuntime`` for the current eval.

        PTC bridges forward this into their ``tool.ainvoke`` calls so
        tools that depend on ``state`` / ``store`` / ``tool_call_id``
        (notably subagent ``task`` tools) see the orchestrator's graph
        context. The middleware calls this immediately before each eval
        and again with ``None`` after.
        """
        self._outer_runtime = runtime

    def _register_tool_bridge(self, camel: str) -> None:
        """Install a host-function bridge for one camel-cased tool name.

        The bridge is async so ``eval_async``'s driving loop can await
        ``tool.ainvoke`` without blocking the event loop. We look the
        tool up through ``self._registered_tools`` on every call so a
        later ``install_tools`` that swaps the underlying object (same
        name, different instance) is picked up without re-registration.
        """
        registered = self._registered_tools

        async def _bridge(raw_input: Any = None) -> str:
            tool = registered.get(camel)
            if tool is None:
                # Shouldn't happen — we only rewrite ``globalThis.tools``
                # with names currently in the map — but if a race causes
                # it, fail loud.
                msg = f"tool '{camel}' not registered"
                raise RuntimeError(msg)
            payload = _normalize_tool_input(raw_input)
            call_id = _synth_tool_call_id(tool.name)
            # Build a ToolCall-shaped input so InjectedToolCallId and the
            # runtime-arg injection in _inject_tool_args_for_ptc fire.
            args = _inject_tool_args_for_ptc(
                tool, payload, self._outer_runtime, call_id
            )
            result = await tool.ainvoke(
                {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"},
            )
            return _coerce_tool_output(result)

        self._ctx.register(f"__tools_{camel}", _bridge, is_async=True)

    def eval_sync(self, code: str) -> EvalOutcome:
        # Both sync and async entry points funnel through ctx.eval_async on
        # the worker loop. Sync ctx.eval can't dispatch async host functions
        # (PTC bridges are is_async=True), so routing sync callers through
        # the async path is required for PTC to work under sync invocation.
        return self._worker.run_sync(self._aeval_async(code))

    async def eval_async(self, code: str) -> EvalOutcome:
        return await self._worker.run_async(self._aeval_async(code))

    async def ainstall_module_scope(self, scope: ModuleScope) -> None:
        """Install a ``ModuleScope`` on the context from the caller's loop."""
        await self._worker.run_async(self._ainstall_module_scope(scope))

    async def _ainstall_module_scope(self, scope: ModuleScope) -> None:
        self._runtime.install(scope)

    async def _aeval_async(self, code: str) -> EvalOutcome:
        """Uses ``ctx.eval_async`` directly.

        Overlapping evals on the same context surface as
        ``ConcurrentEvalError`` (recorded in ``EvalOutcome.error_type``).
        We intentionally do not queue: a model dispatching overlapping
        evals against shared state is almost always a prompting bug,
        and a loud failure is a better signal than silent serialisation.
        """
        outcome = EvalOutcome()
        try:
            value = await self._ctx.eval_async(code, timeout=self._per_call_timeout)
            outcome.result = _stringify(value)
        except MarshalError:
            outcome.result_kind = "handle"
            outcome.result = await self._describe_via_handle_async(code)
        except QJSTimeoutError as e:
            outcome.error_type = "Timeout"
            outcome.error_message = str(e)
        except DeadlockError as e:
            # Top-level Promise never resolved and no async host work in
            # flight. Surface as a distinct error type because the fix
            # is user-level (their JS has an un-resolvable Promise or a
            # sync host fn that should be async); a plain error-type
            # message without context would make this hard to diagnose.
            outcome.error_type = "Deadlock"
            outcome.error_message = str(e)
        except HostCancellationError:
            # JS declined to catch a cancellation — re-raise as
            # CancelledError so asyncio unwinds the caller's task.
            # Do not record anything in ``outcome``; the call is dead.
            raise asyncio.CancelledError from None
        except JSError as e:
            self._record_js_error(outcome, e)
        except ConcurrentEvalError as e:
            outcome.error_type = "ConcurrentEval"
            outcome.error_message = str(e)
        except MemoryLimitError as e:
            outcome.error_type = "OutOfMemory"
            outcome.error_message = str(e)
        outcome.stdout = self._console.drain()
        return outcome

    def _record_js_error(self, outcome: EvalOutcome, e: JSError) -> None:
        # HostError is a JSError subclass; surface it as "HostError"
        # so operators can distinguish a bug in our console bridge
        # from a user-code error.
        if isinstance(e, HostError):
            logger.warning("console-bridge host error", exc_info=e.__cause__)
            outcome.error_type = "HostError"
        else:
            outcome.error_type = e.name
        outcome.error_message = e.message
        outcome.error_stack = e.stack

    async def _describe_via_handle_async(self, code: str) -> str:
        try:
            handle = await self._ctx.eval_handle_async(
                code, timeout=self._per_call_timeout
            )
        except Exception:  # noqa: BLE001 — describe-only path; swallow to placeholder
            return _HANDLE_PLACEHOLDER
        try:
            return _format_handle(handle)
        finally:
            handle.dispose()

    def close(self) -> None:
        self._worker.run_sync(self._aclose())

    async def _aclose(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None


@dataclass
class _SkillInstall:
    """Install-cache entry for one skill on a Runtime.

    Either ``loaded`` is set (install succeeded) or ``error`` is set
    (install failed; subsequent references fail fast without re-hitting
    the backend). Never both.
    """

    loaded: LoadedSkill | None = None
    error: SkillLoadError | None = None


@dataclass
class _Slot:
    """One LangGraph thread's private QuickJS stack: worker + Runtime + REPL.

    Each slot owns an OS thread (via ``ThreadWorker``) and a Runtime. This
    keeps per-conversation JS execution on its own event loop so one
    user's slow computation can't block others. ``installed_skills``
    tracks which skills have already been installed on *this* Runtime —
    the Runtime-independent source cache lives on ``_Registry``.
    """

    worker: ThreadWorker
    runtime: Runtime
    repl: _ThreadREPL
    installed_skills: set[str] = field(default_factory=set)
    last_used: float = 0.0


@dataclass
class _Registry:
    """Per-thread Runtime registry with idle-TTL eviction.

    Each LangGraph ``thread_id`` gets its own ``_Slot`` (worker + Runtime
    + Context). Slots idle longer than ``idle_ttl_sec`` are evicted on
    the next ``get()`` call — eviction is lazy, not via a background
    sweeper.

    A process-wide skill source cache (``_skill_installs``) lives here
    too: source is fetched from the backend once, then installed per-
    Runtime. So when a slot is rebuilt after eviction, its skills re-
    install from cache without another backend roundtrip.
    """

    memory_limit: int
    timeout: float
    capture_console: bool
    idle_ttl_sec: float = 3600.0
    max_active_threads: int | None = None
    _slots: dict[str, _Slot] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Runtime-independent cache of fetched skill source, keyed by skill
    # name. Survives slot eviction. Installs into each Runtime on first
    # reference via that slot; ``_Slot.installed_skills`` dedupes per-
    # Runtime so we only call ``runtime.install`` once.
    _skill_installs: dict[str, _SkillInstall] = field(default_factory=dict)
    _skill_install_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def get(self, thread_id: str) -> _ThreadREPL:
        with self._lock:
            self._evict_stale_locked()
            slot = self._slots.get(thread_id)
            if slot is None:
                self._evict_for_cap_locked()
                slot = self._build_slot_locked(thread_id)
                self._slots[thread_id] = slot
            slot.last_used = time.monotonic()
            return slot.repl

    def _build_slot_locked(self, thread_id: str) -> _Slot:
        name = f"quickjs-worker-{thread_id[:8]}"
        worker = ThreadWorker(name=name)
        runtime = worker.run_sync(self._acreate_runtime())
        repl = _ThreadREPL(
            worker,
            runtime,
            timeout=self.timeout,
            capture_console=self.capture_console,
        )
        return _Slot(worker=worker, runtime=runtime, repl=repl)

    def _evict_stale_locked(self) -> None:
        now = time.monotonic()
        stale = [
            tid
            for tid, slot in self._slots.items()
            if now - slot.last_used > self.idle_ttl_sec
        ]
        for tid in stale:
            self._close_slot(self._slots.pop(tid))

    def _evict_for_cap_locked(self) -> None:
        if self.max_active_threads is None:
            return
        while len(self._slots) >= self.max_active_threads:
            oldest = min(self._slots, key=lambda t: self._slots[t].last_used)
            self._close_slot(self._slots.pop(oldest))

    def _close_slot(self, slot: _Slot) -> None:
        # Best-effort; never block shutdown on a misbehaving runtime.
        with contextlib.suppress(Exception):
            slot.worker.run_sync(_aclose_runtime(slot.runtime))
        slot.worker.close()

    async def _acreate_runtime(self) -> Runtime:
        return Runtime(memory_limit=self.memory_limit)

    async def aensure_skills_installed(
        self,
        referenced: frozenset[str],
        metadata: dict[str, SkillMetadata],
        backend: BackendProtocol,
        repl: _ThreadREPL,
    ) -> list[SkillLoadError]:
        """Install any referenced skills on this slot's Runtime.

        Source is fetched once per skill (cached in ``_skill_installs``),
        then installed per-Runtime. ``_Slot.installed_skills`` dedupes
        so a second eval on the same slot referencing the same skill
        skips the install call.
        """
        errors: list[SkillLoadError] = []
        slot = self._slot_for_repl(repl)
        async with self._skill_install_lock:
            for name in referenced:
                if slot is not None and name in slot.installed_skills:
                    continue
                cached = self._skill_installs.get(name)
                if cached is not None and cached.error is not None:
                    errors.append(cached.error)
                    continue
                loaded = cached.loaded if cached is not None else None
                if loaded is None:
                    meta = metadata.get(name)
                    if meta is None:
                        errors.append(
                            SkillLoadError(
                                f"skill {name!r} referenced but not available "
                                "on this agent"
                            )
                        )
                        continue
                    try:
                        loaded = await aload_skill(meta, backend)
                    except SkillLoadError as exc:
                        self._skill_installs[name] = _SkillInstall(error=exc)
                        errors.append(exc)
                        continue
                    self._skill_installs[name] = _SkillInstall(loaded=loaded)
                await repl.ainstall_module_scope(
                    ModuleScope({loaded.specifier: loaded.scope})
                )
                if slot is not None:
                    slot.installed_skills.add(name)
        return errors

    def _slot_for_repl(self, repl: _ThreadREPL) -> _Slot | None:
        for slot in self._slots.values():
            if slot.repl is repl:
                return slot
        return None

    def close(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                self._close_slot(slot)
            self._slots.clear()
        self._skill_installs.clear()


async def _aclose_runtime(runtime: Runtime) -> None:
    runtime.close()


def format_outcome(
    outcome: EvalOutcome,
    *,
    max_result_chars: int,
) -> str:
    """Render an EvalOutcome as the tool's wire format (see spec §8)."""
    parts: list[str] = []
    if outcome.stdout:
        parts.append(
            f"<stdout>\n{_truncate(outcome.stdout, max_result_chars)}\n</stdout>"
        )
    if outcome.error_type is not None:
        inner = outcome.error_message
        if outcome.error_stack:
            inner = f"{inner}\n{outcome.error_stack}"
        parts.append(
            f'<error type="{_xml_escape(outcome.error_type)}">'
            f"{_xml_escape(_truncate(inner, max_result_chars))}"
            f"</error>"
        )
    else:
        body = outcome.result if outcome.result is not None else "undefined"
        kind_attr = f' kind="{outcome.result_kind}"' if outcome.result_kind else ""
        parts.append(
            f"<result{kind_attr}>{_xml_escape(_truncate(body, max_result_chars))}</result>"
        )
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_TRUNCATE_MARKER.format(n=0)))
    dropped = len(text) - keep
    return text[:keep] + _TRUNCATE_MARKER.format(n=dropped)


def _xml_escape(text: str) -> str:
    # Minimal escape — we emit the tag set we control, so we only need to
    # keep angle brackets from closing our wrapper tags early.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
