"""Ownership gate for the assistant-payload replay hook on ChatOpenAI-shaped providers.

``patched_mimo`` and ``patched_stepfun`` each carried a byte-identical override of
LangChain's ``_get_request_payload``: capture the ``AIMessage`` objects before the
serialiser flattens them, call the parent, then re-attach the fields the payload
dropped. Two owners for one hook order means a fix to the ordering (the capture
*must* happen before the parent call, or the originals are already gone) lands on
one provider and silently not the other.

These tests pin three things the extraction must not silently give back:

1. the owner's behaviour, including the capture-before-parent ordering and that
   the payload it returns is the parent's own object;
2. that both providers route through the owner at runtime — retargeting
   ``AssistantPayloadReplayMixin.payload_restorer`` changes both, which a private
   copy would not — while the restorer stays a per-provider decision, exactly as
   ``assistant_payload_replay.py`` has always drawn that line;
3. a source census over ``models/``: no provider module may restate the hook, and
   the two providers that deliberately keep their own are asserted to still
   differ for the recorded reason. Like
   ``test_search_max_results_single_owner.py``'s deferred-provider list, that
   entry can only shrink.

Not covered: what the providers *do* with the restored field. Streaming and
non-streaming reasoning capture stay under
``tests/test_reasoning_capture_single_owner.py``.
"""

import ast
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from deerflow.models.assistant_payload_replay import (
    AssistantPayloadReplayMixin,
    restore_assistant_payloads,
    restore_reasoning_content,
)
from deerflow.models.patched_mimo import PatchedChatMiMo
from deerflow.models.patched_stepfun import PatchedChatStepFun

OWNER_PATH = Path(restore_assistant_payloads.__code__.co_filename).resolve()
MODELS_DIR = OWNER_PATH.parent

PROVIDERS = {
    "patched_mimo": PatchedChatMiMo,
    "patched_stepfun": PatchedChatStepFun,
}


class _Input:
    """Stand-in for the object ``_convert_input`` returns."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    def to_messages(self) -> list[Any]:
        return self._messages


class _Parent:
    """The base implementation the mixin wraps."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.parent_args: tuple[Any, Any, dict[str, Any]] | None = None

    def _convert_input(self, input_: Any) -> _Input:
        self.calls.append("convert")
        return _Input(input_)

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("parent")
        self.parent_args = (input_, stop, kwargs)
        self.built_payload = {"messages": [{"role": "assistant", "content": "hi"}]}
        return self.built_payload


class _Host(AssistantPayloadReplayMixin, _Parent):
    pass


def _assistant_with_reasoning(reasoning: str) -> AIMessage:
    return AIMessage(content="hi", additional_kwargs={"reasoning_content": reasoning})


# --- 1. the bar ------------------------------------------------------------


def test_owner_returns_the_parent_payload_object() -> None:
    host = _Host()
    original = _assistant_with_reasoning("step-by-step")

    payload = host._get_request_payload([original])

    assert payload is host.built_payload, "the owner must hand back the parent's own payload object, not a rebuild"
    assert host.calls == ["convert", "parent"]


def test_owner_captures_originals_before_calling_the_parent() -> None:
    """The ordering is the whole rule: the parent serialises, so capture must precede it."""
    host = _Host()
    original = _assistant_with_reasoning("ordered capture")

    payload = host._get_request_payload([original])

    assert host.calls[:2] == ["convert", "parent"], f"convert must come first, got {host.calls}"
    # the restored field is only reachable because the AIMessage was read pre-parent
    assert payload["messages"][0]["reasoning_content"] == "ordered capture"


def test_owner_forwards_input_stop_and_kwargs_to_the_parent() -> None:
    host = _Host()
    messages = [_assistant_with_reasoning("r")]

    host._get_request_payload(messages, stop=["\n\ngarbage:"], temperature=0.2)

    forwarded_input, forwarded_stop, forwarded_kwargs = host.parent_args
    assert forwarded_input is messages
    assert forwarded_stop == ["\n\ngarbage:"]
    assert forwarded_kwargs == {"temperature": 0.2}


def test_owner_replays_every_field_the_serialiser_dropped() -> None:
    """Several assistant turns in one payload, so the matching is really exercised."""
    payload_messages = [{"role": "assistant", "content": "hi"} for _ in range(2)]

    class _MultiTurnParent:
        def _convert_input(self, input_: Any) -> _Input:
            return _Input(input_)

        def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
            return {"messages": list(payload_messages)}

    class _MultiTurnHost(AssistantPayloadReplayMixin, _MultiTurnParent):
        pass

    originals = [_assistant_with_reasoning("first"), _assistant_with_reasoning("second")]

    out = _MultiTurnHost()._get_request_payload(originals)

    assert [m.get("reasoning_content") for m in out["messages"]] == ["first", "second"]


def test_the_default_restorer_is_reasoning_content_and_does_not_bind() -> None:
    """Guards the staticmethod wrapper: an unbound callable, reached as-is via self."""
    stored = AssistantPayloadReplayMixin.__dict__["payload_restorer"]

    assert isinstance(stored, staticmethod), "a bare function here would bind and pass an extra self"
    assert stored.__func__ is restore_reasoning_content

    host = _Host()
    assert host.payload_restorer is restore_reasoning_content


def test_a_subclass_can_point_the_restorer_at_another_field() -> None:
    """'Each provider decides which fields to restore' stays true after the fold."""

    def only_additional_kwargs_field(payload_msg: dict[str, Any], orig_msg: AIMessage) -> None:
        payload_msg["custom_field"] = orig_msg.additional_kwargs.get("custom", "")

    class _Other(_Host):
        payload_restorer = staticmethod(only_additional_kwargs_field)

    host = _Other()
    original = AIMessage(content="hi", additional_kwargs={"custom": "kept"})

    payload = host._get_request_payload([original])

    assert payload["messages"][0]["custom_field"] == "kept"
    assert "reasoning_content" not in payload["messages"][0]


# --- 2. both providers route through the one owner --------------------------


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_provider_inherits_the_hook_rather_than_restating_it(provider: str) -> None:
    cls = PROVIDERS[provider]

    assert AssistantPayloadReplayMixin in cls.__mro__
    assert "_get_request_payload" not in vars(cls), f"{provider} grew its own copy back"
    assert cls._get_request_payload is AssistantPayloadReplayMixin._get_request_payload


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_provider_sits_ahead_of_its_chat_model_base(provider: str) -> None:
    """``super()`` must resolve to the base being wrapped, not to the mixin's parent."""
    cls = PROVIDERS[provider]
    mro = list(cls.__mro__)

    assert mro.index(AssistantPayloadReplayMixin) < mro.index(cls.__bases__[1]), f"{provider}: mixin must precede the ChatModel base in the bases tuple"


def test_retargeting_the_owner_changes_both_providers_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def spy(payload_msg: dict[str, Any], orig_msg: AIMessage) -> None:
        seen.append("hit")

    monkeypatch.setattr(AssistantPayloadReplayMixin, "payload_restorer", staticmethod(spy))

    for cls in PROVIDERS.values():
        host = _Host()
        AssistantPayloadReplayMixin._get_request_payload(host, [_assistant_with_reasoning("x")])

    assert seen == ["hit", "hit"], "both faces must read the restorer through the owner"


# --- 3. source census over models/ -----------------------------------------


def _top_level_class_methods(path: Path) -> list[ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    methods: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods.extend(m for m in node.body if isinstance(m, ast.FunctionDef))
    return methods


def test_the_hook_body_exists_in_the_owner_and_the_declared_deferrals() -> None:
    """Recognized by shape, so renaming a copy would not hide it.

    The set is pinned rather than "owner only", because ``patched_openai`` and
    ``patched_deepseek`` legitimately carry a hook matching the same shape; they
    are named below. A provider that re-grows the folded copy would show up here
    and fail, and a provider that lands the follow-up has to leave this set by
    deleting its deferral entry.
    """
    needles = ("to_messages()", "super()._get_request_payload", "restore_assistant_payloads(", "_convert_input")
    matches = []
    for path in sorted(MODELS_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for method in _top_level_class_methods(path):
            text = ast.get_source_segment(source, method) or ""
            if all(needle in text for needle in needles):
                matches.append(f"{path.name}::{method.name}")

    expected = {OWNER_PATH.name + "::_get_request_payload"} | {f"{name}::_get_request_payload" for name in DEFERRED_PROVIDERS}
    assert set(matches) == expected, f"the replay hook escaped or moved: {sorted(matches)}"
    assert "patched_mimo.py::_get_request_payload" not in matches
    assert "patched_stepfun.py::_get_request_payload" not in matches


# Declared deferrals: recorded, asserted, and only removable by someone landing them.
DEFERRED_PROVIDERS = {
    "patched_openai.py": (
        "_restore_tool_call_signatures",  # restores a different field, not reasoning_content
        "def _get_request_payload",
    ),
    "patched_deepseek.py": (
        "deerflow_error_fallback",  # pre-filters the input before the parent call
        "def _get_request_payload",
    ),
}


def test_the_deferred_providers_still_defer_for_the_recorded_reason() -> None:
    for name, markers in DEFERRED_PROVIDERS.items():
        text = (MODELS_DIR / name).read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text, f"{name} no longer matches the reason it was deferred for"
        assert "AssistantPayloadReplayMixin" not in text, f"{name} now uses the owner; drop its deferral entry"


def test_the_owner_still_exports_the_replay_helpers_it_composes() -> None:
    """The mixin must not have narrowed the module's existing surface."""
    import deerflow.models.assistant_payload_replay as owner_module

    for name in ("restore_assistant_payloads", "restore_reasoning_content", "restore_additional_kwargs_field", "AssistantPayloadReplayMixin"):
        assert hasattr(owner_module, name), f"assistant_payload_replay lost {name}"
