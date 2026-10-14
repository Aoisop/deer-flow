"""Helpers for replaying provider-specific assistant message fields.

Several provider adapters need to preserve fields that LangChain stores on the
original ``AIMessage`` but drops when serializing request payloads. This module
keeps the assistant-message matching logic shared while letting each provider
decide which fields to restore.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from langchain_core.messages import AIMessage, BaseMessage

AssistantPayloadRestorer = Callable[[dict[str, Any], AIMessage], None]


def restore_assistant_payloads(
    payload_messages: Sequence[dict[str, Any]],
    original_messages: Sequence[BaseMessage],
    restore: AssistantPayloadRestorer,
) -> None:
    """Restore provider-specific fields onto serialized assistant payloads."""
    ai_messages = [m for m in original_messages if isinstance(m, AIMessage)]
    assistant_payloads = [m for m in payload_messages if m.get("role") == "assistant"]
    used_ai_indexes: set[int] = set()

    for ordinal, payload_msg in enumerate(assistant_payloads):
        ai_msg = _match_ai_message(payload_msg, ai_messages, used_ai_indexes, ordinal)
        if ai_msg is not None:
            restore(payload_msg, ai_msg)


def restore_additional_kwargs_field(payload_msg: dict[str, Any], orig_msg: AIMessage, field_name: str) -> None:
    """Copy a provider-specific ``additional_kwargs`` field onto a payload message."""
    value = orig_msg.additional_kwargs.get(field_name)
    if value is not None:
        payload_msg[field_name] = value


def restore_reasoning_content(payload_msg: dict[str, Any], orig_msg: AIMessage) -> None:
    """Copy provider reasoning content onto a serialized assistant payload."""
    restore_additional_kwargs_field(payload_msg, orig_msg, "reasoning_content")


def _match_ai_message(
    payload_msg: dict[str, Any],
    ai_messages: Sequence[AIMessage],
    used_ai_indexes: set[int],
    fallback_ordinal: int,
) -> AIMessage | None:
    payload_key = _assistant_signature(payload_msg)
    if payload_key is not None:
        matches = [index for index, ai_msg in enumerate(ai_messages) if index not in used_ai_indexes and _ai_signature(ai_msg) == payload_key]
        if len(matches) == 1:
            used_ai_indexes.add(matches[0])
            return ai_messages[matches[0]]

    fallback_index = _next_unused_index_at_or_after(len(ai_messages), used_ai_indexes, fallback_ordinal)
    if fallback_index is not None:
        used_ai_indexes.add(fallback_index)
        return ai_messages[fallback_index]

    return None


def _next_unused_index_at_or_after(count: int, used_ai_indexes: set[int], start: int) -> int | None:
    """Return the next unused AI index at or after ``start``.

    Scanning forward from the payload's ordinal preserves the positional bias of
    the previous behaviour while still recovering when serialization drops or
    reorders messages so the exact ordinal index is already taken. It does not
    wrap to earlier indexes because those messages may be represented by payload
    entries that were already dropped.
    """
    if count == 0 or start >= count:
        return None
    for index in range(start, count):
        if index not in used_ai_indexes:
            return index
    return None


def _assistant_signature(payload_msg: dict[str, Any]) -> tuple[str, str] | None:
    return _signature(
        payload_msg.get("content"),
        _tool_call_ids(payload_msg.get("tool_calls") or []),
    )


def _ai_signature(message: AIMessage) -> tuple[str, str] | None:
    tool_calls = message.tool_calls or message.additional_kwargs.get("tool_calls") or []
    return _signature(message.content, _tool_call_ids(tool_calls))


def _signature(content: Any, tool_call_ids: tuple[str, ...]) -> tuple[str, str] | None:
    if content in (None, "") and not tool_call_ids:
        return None
    return (_stable_repr(content), "|".join(tool_call_ids))


def _stable_repr(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _tool_call_ids(tool_calls: Sequence[Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id:
                ids.append(call_id)
    return tuple(ids)


class AssistantPayloadReplayMixin:
    """``_get_request_payload`` hook that replays dropped assistant fields.

    ``patched_mimo`` and ``patched_stepfun`` each carried a byte-identical
    override of this LangChain hook, doing the same three mechanical steps in the
    same order: capture the ``AIMessage`` objects before the serialiser flattens
    them, let the parent build the payload, then re-attach the fields the payload
    dropped. Which field a provider re-attaches is not this mixin's business --
    that is the boundary this module has always drawn ("letting each provider
    decide which fields to restore") -- so it is a ``payload_restorer`` the
    subclass can point anywhere, defaulting to ``restore_reasoning_content``,
    which is the restorer both copies already used.

    Place it **before** the ChatModel base in the bases tuple so ``super()``
    resolves to the base implementation being wrapped:

        class PatchedChatMiMo(AssistantPayloadReplayMixin, ChatOpenAI): ...

    Providers whose hook does something other than replay stay local:
    ``patched_deepseek`` filters ``deerflow_error_fallback`` messages out of the
    input *before* calling the parent, and ``patched_openai`` restores
    ``thought_signature`` on tool-call objects while five in-flight PRs are
    editing that same method.
    """

    # Wrapped in staticmethod on purpose: a bare function placed on a class would bind
    # as an instance method, and self.payload_restorer would then hand the restorer
    # an extra leading ``self`` argument. Subclasses override the same way.
    payload_restorer: ClassVar[AssistantPayloadRestorer] = staticmethod(restore_reasoning_content)

    def _get_request_payload(
        self: Any,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Rebuild the parent payload, then replay assistant fields onto it."""
        original_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        restore_assistant_payloads(payload.get("messages", []), original_messages, self.payload_restorer)
        return payload
