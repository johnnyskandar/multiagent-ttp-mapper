"""Pretty printer for the shared message audit log (rubric R12)."""

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_MAX_CONTENT_CHARS = 1500


def _truncate(text: str, limit: int = _MAX_CONTENT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


def _header(label: str, msg: BaseMessage) -> str:
    name = getattr(msg, "name", None)
    return f"[{label}]" + (f" (agent: {name})" if name else "")


def format_message(msg: BaseMessage) -> str:
    """Return a readable multi-line string for one message, labeled by its type."""
    content = _truncate(_content_to_str(msg.content))
    if isinstance(msg, SystemMessage):
        return f"{_header('SystemMessage', msg)}\n{content}"
    if isinstance(msg, HumanMessage):
        return f"{_header('HumanMessage', msg)}\n{content}"
    if isinstance(msg, ToolMessage):
        return (
            f"[ToolMessage] tool={msg.name} call_id={msg.tool_call_id}"
            f" status={getattr(msg, 'status', 'success')}\n{content}"
        )
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            lines = [_header("AIMessage with tool calls", msg)]
            if content:
                lines.append(content)
            for call in msg.tool_calls:
                args = json.dumps(call.get("args", {}), ensure_ascii=False, default=str)
                lines.append(f"  -> call {call.get('name')} id={call.get('id')} args={_truncate(args, 500)}")
            return "\n".join(lines)
        return f"{_header('AIMessage', msg)}\n{content}"
    return f"[{type(msg).__name__}]\n{content}"


def print_messages(messages: list[BaseMessage]) -> None:
    """Print every message in order with a clear type label and separator."""
    for i, msg in enumerate(messages):
        print(f"--- {i:02d} " + "-" * 60)
        print(format_message(msg))
    print("-" * 67)
