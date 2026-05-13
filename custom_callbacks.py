import json
import os
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
SYSTEM_PROMPT_DEFAULT = "Reply concisely."

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
