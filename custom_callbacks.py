import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

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

# Матчит ТОЛЬКО закрытые пары <reasoning>…</reasoning> — незакрытые блоки остаются как есть.
_REASONING_RE = re.compile(
    rf"<{REASONING_TAG}\b[^>]*>.*?</{REASONING_TAG}\s*>\s*",
    re.DOTALL | re.IGNORECASE,
)


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
        if call_type not in ("completion", "text_completion"):
            return data
        if not REASONING_INSTRUCTION:
            return data

        messages = data.get("messages") or []

        # Find the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        # Prepare the instruction block with system-reminder tags
        instruction_block = {
            "type": "text",
            "text": f"<system-reminder>{REASONING_INSTRUCTION}</system-reminder>\n\n"
        }

        if last_user_idx is not None:
            # Last user message exists: prepend instruction to it
            user_msg = messages[last_user_idx]
            content = user_msg.get("content")

            if isinstance(content, str):
                # Convert string to content array
                user_msg["content"] = [
                    instruction_block,
                    {"type": "text", "text": content}
                ]
            elif isinstance(content, list):
                # Already a content array: prepend instruction block
                user_msg["content"].insert(0, instruction_block)
            else:
                # Unexpected content type, leave unchanged
                pass
        else:
            # No user message: create one with just the instruction
            messages.append({
                "role": "user",
                "content": [instruction_block]
            })

        data["messages"] = messages
        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        for choice in getattr(response, "choices", None) or []:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                message.content = _REASONING_RE.sub("", content)
        return response


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
