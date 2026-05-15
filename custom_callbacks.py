import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable, Literal, Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth


# =====================================================================
# Модификация запросов (async_pre_call_hook)
# =====================================================================

# Системный промт. Встаёт первым; если system-сообщение уже есть — наш текст
# дописывается в НАЧАЛО его content. Пустая строка — функция выключена.
SYSTEM_PROMPT_DEFAULT = ""

# Текст, дописываемый в КОНЕЦ последнего user-сообщения.
# Пустая строка — функция выключена.
USER_PROMPT_APPEND = ""


class RequestModifier(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[dict]:
        if call_type not in ("completion", "text_completion"):
            return data

        messages = data.get("messages") or []

        if SYSTEM_PROMPT_DEFAULT:
            sys_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "system"),
                None,
            )
            if sys_idx is None:
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_DEFAULT})
            else:
                existing = messages[sys_idx].get("content") or ""
                separator = "\n\n" if existing else ""
                messages[sys_idx]["content"] = SYSTEM_PROMPT_DEFAULT + separator + existing

        if USER_PROMPT_APPEND:
            for m in reversed(messages):
                if m.get("role") == "user":
                    m["content"] = (m.get("content") or "") + USER_PROMPT_APPEND
                    break

        data["messages"] = messages
        return data


request_modifier = RequestModifier()


# =====================================================================
# Удаление reasoning-тега из ответа модели
# (async_pre_call_hook добавляет инструкцию, async_post_call_success_hook чистит)
# =====================================================================

# Имя XML-тега, в который просим модель оборачивать reasoning.
REASONING_TAG = "reasoning"

# Текст, добавляемый в КОНЕЦ system-сообщения. Пустая строка — функция выключена.
REASONING_INSTRUCTION = (
    f"You MUST begin every response with an internal reasoning block wrapped in "
    f"<{REASONING_TAG}>...</{REASONING_TAG}> XML tags, placed BEFORE your final answer. "
    "This is not optional — omitting the block is a protocol violation.\n\n"
    "Inside the block, think step by step: restate the goal in your own words, list "
    "the relevant facts/constraints/assumptions, enumerate the options or sub-steps, "
    "and verify the conclusion before writing the final answer.\n\n"
    "Length policy:\n"
    "- Absolute minimum (even for trivial questions like greetings or one-word answers): "
    "  at least 3 substantive sentences (~50 words) covering goal, approach, and a sanity check.\n"
    "- Scale the depth up with task complexity. For multi-step problems, code, math, "
    "  ambiguous requirements, or anything requiring tradeoff analysis — expand the block "
    "  proportionally. There is no upper bound; prefer deeper, more thorough reasoning "
    "  over shallow.\n"
    "- For genuinely hard problems, aim for hundreds of words and explicitly consider "
    "  edge cases, alternative interpretations, and failure modes.\n\n"
    "The reasoning block will be stripped from the response before it reaches the user, "
    "so write it for yourself, not for the user. After </{tag}> produce the actual answer "
    "the user will see."
).format(tag=REASONING_TAG)

def _xml_tag_re(tag: str) -> re.Pattern:
    return re.compile(
        rf"<{re.escape(tag)}\b[^>]*>.*?</{re.escape(tag)}\s*>",
        re.DOTALL | re.IGNORECASE,
    )


def _partial_tag_suffix_len(text: str, marker: str) -> int:
    lower_text = text.lower()
    max_len = min(len(lower_text), len(marker))
    for size in range(max_len, 0, -1):
        if marker.startswith(lower_text[-size:]):
            return size
    return 0


class ReasoningStreamFilter:
    def __init__(self, tag: str) -> None:
        escaped_tag = re.escape(tag)
        self._opening_re = re.compile(
            rf"<{escaped_tag}\b[^>]*>",
            re.IGNORECASE,
        )
        self._closing_re = re.compile(
            rf"</{escaped_tag}\s*>",
            re.IGNORECASE,
        )
        self._opening_marker = f"<{tag.lower()}"
        self._closing_marker = f"</{tag.lower()}"
        self._buffer = ""
        self._inside_reasoning = False
        self._drop_leading_ws = False
        self._emitted_visible = False

    def feed(self, text: str) -> str:
        if not text:
            return ""

        self._buffer += text
        visible_parts = []

        while self._buffer:
            if self._inside_reasoning:
                closing_match = self._closing_re.search(self._buffer)
                if closing_match is None:
                    self._buffer = self._inside_pending_suffix()
                    break

                self._buffer = self._buffer[closing_match.end():]
                self._inside_reasoning = False
                self._drop_leading_ws = not self._emitted_visible
                continue

            if self._drop_leading_ws:
                self._buffer = self._buffer.lstrip()
                if not self._buffer:
                    break
                self._drop_leading_ws = False

            opening_match = self._opening_re.search(self._buffer)
            if opening_match is None:
                visible, self._buffer = self._outside_visible_prefix()
                visible_parts.append(visible)
                break

            visible = self._buffer[:opening_match.start()]
            visible_parts.append(visible)
            if visible:
                self._emitted_visible = True
            self._buffer = self._buffer[opening_match.end():]
            self._inside_reasoning = True

        visible = "".join(visible_parts)
        if visible:
            self._emitted_visible = True
        return visible

    def flush(self) -> str:
        if self._inside_reasoning:
            self._buffer = ""
            self._inside_reasoning = False
            self._drop_leading_ws = False
            return ""

        if self._drop_leading_ws:
            self._buffer = self._buffer.lstrip()
            self._drop_leading_ws = False

        visible = self._buffer
        self._buffer = ""
        if visible:
            self._emitted_visible = True
        return visible

    def _inside_pending_suffix(self) -> str:
        lower_buffer = self._buffer.lower()
        marker_idx = lower_buffer.rfind(self._closing_marker)
        if marker_idx != -1:
            return self._buffer[marker_idx:]

        keep_len = _partial_tag_suffix_len(self._buffer, self._closing_marker)
        if keep_len:
            return self._buffer[-keep_len:]
        return ""

    def _outside_visible_prefix(self) -> tuple[str, str]:
        lower_buffer = self._buffer.lower()
        marker_idx = lower_buffer.rfind(self._opening_marker)
        if marker_idx != -1:
            return self._buffer[:marker_idx], self._buffer[marker_idx:]

        keep_len = _partial_tag_suffix_len(self._buffer, self._opening_marker)
        if keep_len:
            return self._buffer[:-keep_len], self._buffer[-keep_len:]
        return self._buffer, ""


_REASONING_RE = _xml_tag_re(REASONING_TAG)
_REASONING_WITH_SEPARATOR_RE = re.compile(
    rf"<{re.escape(REASONING_TAG)}\b[^>]*>.*?</{re.escape(REASONING_TAG)}\s*>"
    r"[ \t]*(?:\r?\n)+\s*",
    re.DOTALL | re.IGNORECASE,
)
_UNCLOSED_REASONING_RE = re.compile(
    rf"<{re.escape(REASONING_TAG)}\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)
_SYSTEM_REMINDER_RE = _xml_tag_re("system-reminder")
_REASONING_OPENING_START_RE = re.compile(
    rf"\A\s*<{re.escape(REASONING_TAG)}\b[^>]*>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_reasoning_text(content: str) -> str:
    stripped = _REASONING_WITH_SEPARATOR_RE.sub("", content)
    stripped = _REASONING_RE.sub("", stripped)
    if _REASONING_OPENING_START_RE.match(content):
        stripped = stripped.lstrip()
    return _UNCLOSED_REASONING_RE.sub("", stripped)


# Constant block prepended to user messages (built once, used many times)
_SYSTEM_REMINDER_BLOCK_TEXT = f"<system-reminder>{REASONING_INSTRUCTION}</system-reminder>\n\n"
_INSTRUCTION_BLOCK = {"type": "text", "text": _SYSTEM_REMINDER_BLOCK_TEXT}


def _stream_choice_index(choice: Any, fallback: int) -> int:
    if isinstance(choice, dict):
        index = choice.get("index", fallback)
    else:
        index = getattr(choice, "index", fallback)
    if isinstance(index, int):
        return index
    return fallback


def _get_chunk_choices(chunk: Any) -> Iterable[Any]:
    if isinstance(chunk, dict):
        choices = chunk.get("choices")
    else:
        choices = getattr(chunk, "choices", None)
    return choices or []


def _get_delta_content(choice: Any) -> Optional[str]:
    if isinstance(choice, dict):
        delta = choice.get("delta")
    else:
        delta = getattr(choice, "delta", None)
    if isinstance(delta, dict):
        content = delta.get("content")
    else:
        content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    return None


def _can_set_delta_content(choice: Any) -> bool:
    if isinstance(choice, dict):
        return isinstance(choice.get("delta"), dict)
    return getattr(choice, "delta", None) is not None


def _set_delta_content(choice: Any, content: str) -> None:
    if isinstance(choice, dict):
        delta = choice.get("delta")
    else:
        delta = getattr(choice, "delta", None)
    if delta is None:
        return
    if isinstance(delta, dict):
        delta["content"] = content
        return
    setattr(delta, "content", content)


def _try_set_delta_content(choice: Any, content: str) -> bool:
    try:
        _set_delta_content(choice, content)
    except Exception:
        return False
    return True


def _try_ensure_dict_delta_content(choice: Any, content: str) -> bool:
    if not isinstance(choice, dict):
        return False
    delta = choice.setdefault("delta", {})
    if not isinstance(delta, dict):
        return False
    delta["content"] = content
    return True


def _get_finish_reason(choice: Any) -> Any:
    if isinstance(choice, dict):
        return choice.get("finish_reason")
    return getattr(choice, "finish_reason", None)


def _get_response_choices(response: Any) -> Iterable[Any]:
    if isinstance(response, dict):
        choices = response.get("choices")
    else:
        choices = getattr(response, "choices", None)
    return choices or []


def _get_message_content(choice: Any) -> Optional[str]:
    if isinstance(choice, dict):
        message = choice.get("message")
    else:
        message = getattr(choice, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return None


def _set_message_content(choice: Any, content: str) -> None:
    if isinstance(choice, dict):
        message = choice.get("message")
    else:
        message = getattr(choice, "message", None)
    if message is None:
        return
    if isinstance(message, dict):
        message["content"] = content
        return
    setattr(message, "content", content)


class ReasoningStripper(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[dict]:
        if not REASONING_INSTRUCTION:
            return data
        if call_type not in ("completion", "text_completion"):
            return data

        messages = data.get("messages") or []

        # Find the last user message
        last_user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1)
             if messages[i].get("role") == "user"),
            None,
        )

        if last_user_idx is not None:
            user_msg = messages[last_user_idx]
            content = user_msg.get("content")

            if isinstance(content, str):
                cleaned = _SYSTEM_REMINDER_RE.sub("", content).strip()
                # Convert to content array
                user_msg["content"] = [
                    _INSTRUCTION_BLOCK,
                    {"type": "text", "text": cleaned}
                ]
            elif isinstance(content, list):
                # Remove any existing system-reminder blocks from content array
                filtered_content = []
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        cleaned = _SYSTEM_REMINDER_RE.sub("", text).strip()
                        if cleaned != text.strip():
                            if cleaned:
                                filtered_content.append({"type": "text", "text": cleaned})
                        elif text:
                            filtered_content.append(block)
                    else:
                        filtered_content.append(block)

                # Prepend instruction block
                user_msg["content"] = [_INSTRUCTION_BLOCK] + filtered_content
        else:
            # No user message: create one with just the instruction
            messages.append({
                "role": "user",
                "content": [_INSTRUCTION_BLOCK]
            })

        data["messages"] = messages
        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        for choice in _get_response_choices(response):
            content = _get_message_content(choice)
            if isinstance(content, str):
                _set_message_content(choice, _strip_reasoning_text(content))
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        filters: dict[int, ReasoningStreamFilter] = {}

        async for chunk in response:
            choices = _get_chunk_choices(chunk)
            for position, choice in enumerate(choices):
                try:
                    choice_index = _stream_choice_index(choice, position)
                    content = _get_delta_content(choice)
                    can_write_content = _can_set_delta_content(choice)
                except Exception:
                    continue

                stream_filter = filters.setdefault(
                    choice_index,
                    ReasoningStreamFilter(REASONING_TAG),
                )

                if content is not None:
                    if not can_write_content:
                        continue
                    if not _try_set_delta_content(choice, ""):
                        continue
                    filtered_content = stream_filter.feed(content)
                    _try_set_delta_content(choice, filtered_content)

                try:
                    finish_reason = _get_finish_reason(choice)
                except Exception:
                    continue

                if finish_reason is not None:
                    current = _get_delta_content(choice) or ""
                    tail = stream_filter.flush()
                    if tail:
                        if can_write_content:
                            _try_set_delta_content(choice, current + tail)
                        else:
                            _try_ensure_dict_delta_content(choice, current + tail)
                    filters.pop(choice_index, None)

            yield chunk


reasoning_stripper = ReasoningStripper()


# =====================================================================
# JSONL-логирование (async_log_success_event / async_log_failure_event)
# =====================================================================

LOG_DIR = Path(os.getenv("PROXY_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "requests.jsonl"


class JsonlLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, "success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, "failure")

    def _write(self, kwargs: dict, status: str) -> None:
        entry = dict(kwargs.get("standard_logging_object") or {})
        entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
        entry.setdefault("status", status)
        entry.setdefault("model", kwargs.get("model"))
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


jsonl_logger = JsonlLogger()
