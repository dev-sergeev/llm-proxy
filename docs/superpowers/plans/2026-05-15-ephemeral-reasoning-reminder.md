# Ephemeral Reasoning Reminder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the proxy inject a fresh reasoning reminder at the active end of every completion request, including tool-loop requests that end with `assistant` or `tool`.

**Architecture:** Keep `RequestModifier` as the only request-mutating callback. Change reasoning injection so stale reminders are removed from historical user messages, then either refresh the final `user` message or append a synthetic trailing `user` reminder. Add a bounded audit trail so `logs/requests.jsonl` can prove the upstream response contained `<reasoning>` before `ReasoningStripper` removed it from client-visible output.

**Tech Stack:** Python 3.12, LiteLLM proxy callbacks, pytest, stdlib `urllib` e2e tests, Qwen Code CLI for the final real-agent smoke test.

---

## File Structure

- `custom_callbacks.py` owns request mutation, response stripping, and JSONL logging. It will receive focused helper changes only; no callback responsibilities move to new files.
- `tests/test_custom_callbacks.py` covers request mutation and audit logging at unit level.
- `tests/test_reasoning_proxy_e2e.py` launches a mock upstream and LiteLLM proxy, then verifies tool-loop request shape, client-visible stripping, and JSONL audit fields.
- `README.md` documents the trailing-reminder behavior and the new log fields.

## Task 1: Request Mutation Regression Tests

**Files:**
- Modify: `tests/test_custom_callbacks.py`
- Test: `tests/test_custom_callbacks.py`

- [ ] **Step 1: Write failing tests for non-user final messages**

Add these tests inside `class TestRequestModifierReasoningInjection`, after `test_preserves_array_text_whitespace_when_cleaning_existing_reminder`:

```python
    @pytest.mark.asyncio
    async def test_appends_synthetic_user_reminder_when_final_message_is_tool(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "<system-reminder>Old instruction</system-reminder>\n\nUse the tool.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "tool output",
                },
            ]
        }

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        assert len(result["messages"]) == 4
        assert result["messages"][0]["content"] == "Use the tool."
        assert result["messages"][2] == data["messages"][2]
        tail = result["messages"][-1]
        assert tail["role"] == "user"
        assert isinstance(tail["content"], str)
        assert tail["content"].startswith("<system-reminder>")
        assert REASONING_INSTRUCTION in tail["content"]
        assert tail["content"].rstrip().endswith("</system-reminder>")
        assert "tool output" not in tail["content"]

    @pytest.mark.asyncio
    async def test_appends_synthetic_user_reminder_when_final_message_is_assistant(self):
        data = {
            "messages": [
                {"role": "user", "content": "Call a tool if needed."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "list_files", "arguments": "{}"},
                        }
                    ],
                },
            ]
        }

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        assert len(result["messages"]) == 3
        assert result["messages"][0]["content"] == "Call a tool if needed."
        assert result["messages"][1] == data["messages"][1]
        tail = result["messages"][-1]
        assert tail["role"] == "user"
        assert tail["content"].startswith("<system-reminder>")
        assert REASONING_INSTRUCTION in tail["content"]

    @pytest.mark.asyncio
    async def test_cleans_stale_reminders_from_all_user_messages_before_appending_tail(
        self,
    ):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "<system-reminder>Old A</system-reminder>\n\nFirst request",
                },
                {"role": "assistant", "content": "Intermediate answer"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>Old B</system-reminder>\n\nSecond request",
                        },
                        {"type": "text", "text": "Keep this block"},
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "tool result",
                },
            ]
        }

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        assert result["messages"][0]["content"] == "First request"
        assert result["messages"][2]["content"] == [
            {"type": "text", "text": "Second request"},
            {"type": "text", "text": "Keep this block"},
        ]
        assert result["messages"][-1]["role"] == "user"
        assert result["messages"][-1]["content"].count("<system-reminder>") == 1
        joined = repr(result["messages"])
        assert "Old A" not in joined
        assert "Old B" not in joined

    @pytest.mark.asyncio
    async def test_appends_synthetic_user_reminder_for_empty_messages(self):
        data = {"messages": []}

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        assert result["messages"] == [
            {
                "role": "user",
                "content": result["messages"][0]["content"],
            }
        ]
        assert result["messages"][0]["content"].startswith("<system-reminder>")
        assert REASONING_INSTRUCTION in result["messages"][0]["content"]
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest \
  tests/test_custom_callbacks.py::TestRequestModifierReasoningInjection::test_appends_synthetic_user_reminder_when_final_message_is_tool \
  tests/test_custom_callbacks.py::TestRequestModifierReasoningInjection::test_appends_synthetic_user_reminder_when_final_message_is_assistant \
  tests/test_custom_callbacks.py::TestRequestModifierReasoningInjection::test_cleans_stale_reminders_from_all_user_messages_before_appending_tail \
  tests/test_custom_callbacks.py::TestRequestModifierReasoningInjection::test_appends_synthetic_user_reminder_for_empty_messages \
  -v
```

Expected: the first three tests fail because the current implementation mutates the last historical `user` instead of appending a trailing reminder. The empty-messages test may already pass; keep it because it locks the desired behavior.

## Task 2: Implement Ephemeral Trailing Reminder

**Files:**
- Modify: `custom_callbacks.py`
- Test: `tests/test_custom_callbacks.py`

- [ ] **Step 1: Replace the request-side helper block**

In `custom_callbacks.py`, replace the current `_inject_reasoning_reminder_into_blocks` and `_inject_reasoning_reminder` helper area with this complete block:

```python
def _clean_system_reminder_blocks(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered_content = []
    for block in content:
        if not isinstance(block, dict):
            filtered_content.append(block)
            continue
        if block.get("type") != "text":
            filtered_content.append(block)
            continue

        cleaned = _clean_system_reminder_text(block.get("text", ""))
        if cleaned:
            filtered_content.append({**block, "text": cleaned})

    return filtered_content


def _inject_reasoning_reminder_into_blocks(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [_INSTRUCTION_BLOCK.copy()] + _clean_system_reminder_blocks(content)


def _clean_reasoning_reminder_from_user_message(user_msg: dict[str, Any]) -> None:
    content = user_msg.get("content")
    if isinstance(content, list):
        user_msg["content"] = _clean_system_reminder_blocks(content)
    elif isinstance(content, str):
        user_msg["content"] = _clean_system_reminder_text(content)


def _clean_existing_user_reasoning_reminders(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if message.get("role") == "user":
            _clean_reasoning_reminder_from_user_message(message)


def _append_reasoning_reminder_user_message(messages: list[dict[str, Any]]) -> None:
    messages.append(
        {
            "role": "user",
            "content": _SYSTEM_REMINDER_BLOCK_TEXT.rstrip(),
        }
    )


def _inject_reasoning_reminder(messages: list[dict[str, Any]]) -> None:
    _clean_existing_user_reasoning_reminders(messages)

    if not messages:
        _append_reasoning_reminder_user_message(messages)
        return

    tail = messages[-1]
    if tail.get("role") != "user":
        _append_reasoning_reminder_user_message(messages)
        return

    content = tail.get("content")
    if isinstance(content, list):
        tail["content"] = _inject_reasoning_reminder_into_blocks(content)
    elif isinstance(content, str):
        tail["content"] = _inject_reasoning_reminder_into_string(content)
    else:
        _append_reasoning_reminder_user_message(messages)
```

- [ ] **Step 2: Run the focused request tests**

Run:

```bash
pytest tests/test_custom_callbacks.py::TestRequestModifierReasoningInjection -v
```

Expected: all tests in `TestRequestModifierReasoningInjection` pass.

- [ ] **Step 3: Commit request mutation changes**

Run:

```bash
git add custom_callbacks.py tests/test_custom_callbacks.py
git commit -m "fix: append reasoning reminder after tool messages"
```

## Task 3: Audit Logging Unit Tests

**Files:**
- Modify: `tests/test_custom_callbacks.py`
- Test: `tests/test_custom_callbacks.py`

- [ ] **Step 1: Extend imports for JSONL audit tests**

At the top of `tests/test_custom_callbacks.py`, add `json` and the audit-related imports:

```python
import json
from copy import deepcopy

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_callbacks import (
    JsonlLogger,
    RequestModifier,
    REASONING_INSTRUCTION,
    ReasoningStreamFilter,
    ReasoningStripper,
    _REASONING_AUDIT_BY_CALL_ID,
)
```

- [ ] **Step 2: Add failing audit tests**

Add this class after `class TestReasoningStripperRequestPath`:

```python
class TestReasoningAuditLogging:
    def setup_method(self):
        _REASONING_AUDIT_BY_CALL_ID.clear()

    def teardown_method(self):
        _REASONING_AUDIT_BY_CALL_ID.clear()

    @pytest.mark.asyncio
    async def test_non_streaming_stripper_records_reasoning_observed(self):
        stripper = ReasoningStripper()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "<reasoning>hidden notes</reasoning>Visible"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        await stripper.async_post_call_success_hook(
            data={"litellm_call_id": "call-non-stream"},
            user_api_key_dict=None,
            response=mock_response,
        )

        audit = _REASONING_AUDIT_BY_CALL_ID["call-non-stream"]
        assert audit["proxy_reasoning_observed"] is True
        assert audit["proxy_reasoning_response_mode"] == "non_stream"
        assert "<reasoning>hidden notes</reasoning>Visible" in audit[
            "proxy_raw_response_preview"
        ]
        assert mock_response.choices[0].message.content == "Visible"

    @pytest.mark.asyncio
    async def test_jsonl_logger_merges_and_removes_reasoning_audit(
        self,
        tmp_path,
        monkeypatch,
    ):
        log_file = tmp_path / "requests.jsonl"
        monkeypatch.setattr("custom_callbacks.LOG_FILE", log_file)
        _REASONING_AUDIT_BY_CALL_ID["call-log"] = {
            "proxy_reasoning_observed": True,
            "proxy_reasoning_response_mode": "non_stream",
            "proxy_raw_response_preview": "<reasoning>x</reasoning>Visible",
            "proxy_modified_messages": [
                {"role": "user", "content": "<system-reminder>x</system-reminder>"}
            ],
        }

        logger = JsonlLogger()
        await logger.async_log_success_event(
            kwargs={
                "standard_logging_object": {
                    "litellm_call_id": "call-log",
                    "messages": [{"role": "user", "content": "original"}],
                    "response": {"choices": []},
                }
            },
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        entry = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert entry["proxy_reasoning_observed"] is True
        assert entry["proxy_reasoning_response_mode"] == "non_stream"
        assert entry["proxy_raw_response_preview"] == "<reasoning>x</reasoning>Visible"
        assert entry["proxy_modified_messages"] == [
            {"role": "user", "content": "<system-reminder>x</system-reminder>"}
        ]
        assert "call-log" not in _REASONING_AUDIT_BY_CALL_ID

    def test_stream_filter_marks_when_reasoning_tag_is_seen(self):
        stream_filter = ReasoningStreamFilter("reasoning")

        assert stream_filter.saw_reasoning is False
        assert stream_filter.feed("<rea") == ""
        assert stream_filter.saw_reasoning is False
        assert stream_filter.feed("soning>hidden</reasoning>Visible") == "Visible"
        assert stream_filter.saw_reasoning is True
```

- [ ] **Step 3: Run audit tests and verify they fail**

Run:

```bash
pytest tests/test_custom_callbacks.py::TestReasoningAuditLogging -v
```

Expected: FAIL during import because `_REASONING_AUDIT_BY_CALL_ID` is not available yet. After that symbol exists, the stream-filter assertion should fail until `ReasoningStreamFilter.saw_reasoning` is implemented.

## Task 4: Implement Bounded Reasoning Audit Fields

**Files:**
- Modify: `custom_callbacks.py`
- Test: `tests/test_custom_callbacks.py`

- [ ] **Step 1: Add `deepcopy` import**

Modify the imports at the top of `custom_callbacks.py`:

```python
from copy import deepcopy
import json
import os
import re
```

- [ ] **Step 2: Add audit globals and helpers**

Add this block after `_REASONING_OPENING_START_RE`:

```python
_REASONING_OPENING_RE = re.compile(
    rf"<{re.escape(REASONING_TAG)}\b[^>]*>",
    re.DOTALL | re.IGNORECASE,
)
_RAW_RESPONSE_PREVIEW_CHARS = int(
    os.getenv("PROXY_RAW_RESPONSE_PREVIEW_CHARS", "1000")
)
_REASONING_AUDIT_BY_CALL_ID: dict[str, dict[str, Any]] = {}


def _call_id_from_data(data: dict[str, Any]) -> Optional[str]:
    call_id = data.get("litellm_call_id")
    if isinstance(call_id, str) and call_id:
        return call_id

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata_call_id = metadata.get("litellm_call_id")
        if isinstance(metadata_call_id, str) and metadata_call_id:
            return metadata_call_id

    return None


def _update_reasoning_audit(data: dict[str, Any], **fields: Any) -> None:
    call_id = _call_id_from_data(data)
    if not call_id:
        return

    audit = _REASONING_AUDIT_BY_CALL_ID.setdefault(call_id, {})
    audit.update(fields)


def _message_has_system_reminder(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return _SYSTEM_REMINDER_RE.search(content) is not None
    if isinstance(content, list):
        return any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and _SYSTEM_REMINDER_RE.search(block.get("text", "")) is not None
            for block in content
        )
    return False


def _record_request_reasoning_audit(
    data: dict[str, Any],
    messages: list[dict[str, Any]],
) -> None:
    tail = messages[-1] if messages else {}
    _update_reasoning_audit(
        data,
        proxy_modified_messages=deepcopy(messages),
        proxy_reasoning_reminder_tail_role=tail.get("role"),
        proxy_reasoning_reminder_tail_injected=_message_has_system_reminder(tail),
    )


def _raw_response_preview(text: str) -> str:
    return text[:_RAW_RESPONSE_PREVIEW_CHARS]


def _record_response_reasoning_audit(
    data: dict[str, Any],
    raw_text: str,
    mode: str,
) -> None:
    if not raw_text:
        return

    _update_reasoning_audit(
        data,
        proxy_reasoning_observed=_REASONING_OPENING_RE.search(raw_text) is not None,
        proxy_reasoning_response_mode=mode,
        proxy_raw_response_preview=_raw_response_preview(raw_text),
    )
```

- [ ] **Step 3: Record mutated messages from `RequestModifier`**

In `RequestModifier.async_pre_call_hook`, replace the final block:

```python
        data["messages"] = messages
        return data
```

with:

```python
        data["messages"] = messages
        _record_request_reasoning_audit(data, messages)
        return data
```

- [ ] **Step 4: Add `saw_reasoning` to the streaming filter**

In `ReasoningStreamFilter.__init__`, add:

```python
        self.saw_reasoning = False
```

In `ReasoningStreamFilter.feed`, immediately after `opening_match` is found and before the buffer is advanced, add:

```python
            self.saw_reasoning = True
```

- [ ] **Step 5: Record raw non-streaming response preview before stripping**

In `ReasoningStripper.async_post_call_success_hook`, replace the method body with:

```python
        raw_parts = []
        for choice in _get_response_choices(response):
            content = _get_message_content(choice)
            if isinstance(content, str):
                raw_parts.append(content)
                _set_message_content(choice, _strip_reasoning_text(content))

        _record_response_reasoning_audit(data, "".join(raw_parts), "non_stream")
        return response
```

- [ ] **Step 6: Record streaming response preview when reasoning is seen**

In `ReasoningStripper.async_post_call_streaming_iterator_hook`, initialize a preview accumulator before the `async for` loop:

```python
        raw_preview_parts: list[str] = []
```

Inside the `if content is not None:` block, before `filtered_content = stream_filter.feed(content)`, add:

```python
                    if len("".join(raw_preview_parts)) < _RAW_RESPONSE_PREVIEW_CHARS:
                        raw_preview_parts.append(content)
```

After `filtered_content = stream_filter.feed(content)`, add:

```python
                    if stream_filter.saw_reasoning:
                        _record_response_reasoning_audit(
                            request_data,
                            "".join(raw_preview_parts),
                            "stream",
                        )
```

- [ ] **Step 7: Merge audit fields into JSONL entries**

In `JsonlLogger._write`, after:

```python
        entry = dict(kwargs.get("standard_logging_object") or {})
```

add:

```python
        call_id = entry.get("litellm_call_id") or kwargs.get("litellm_call_id")
        if isinstance(call_id, str):
            entry.update(_REASONING_AUDIT_BY_CALL_ID.pop(call_id, {}))
```

- [ ] **Step 8: Run audit tests**

Run:

```bash
pytest tests/test_custom_callbacks.py::TestReasoningAuditLogging -v
```

Expected: PASS.

- [ ] **Step 9: Run all callback unit tests**

Run:

```bash
pytest tests/test_custom_callbacks.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit audit logging changes**

Run:

```bash
git add custom_callbacks.py tests/test_custom_callbacks.py
git commit -m "feat: log reasoning audit evidence"
```

## Task 5: Proxy E2E Tool-Loop And Log Verification

**Files:**
- Modify: `tests/test_reasoning_proxy_e2e.py`
- Test: `tests/test_reasoning_proxy_e2e.py`

- [ ] **Step 1: Add JSONL wait helper**

Add this helper after `_collect_stream_text`:

```python
def _wait_for_jsonl_entries(path: Path, count: int, timeout: float = 10.0):
    deadline = time.time() + timeout
    last_entries = []
    while time.time() < deadline:
        if path.exists():
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            last_entries = [json.loads(line) for line in lines]
            if len(last_entries) >= count:
                return last_entries
        time.sleep(0.1)

    raise AssertionError(
        f"expected at least {count} JSONL log entries in {path}, got {len(last_entries)}"
    )
```

- [ ] **Step 2: Enable isolated JSONL logging in the e2e config**

Inside `test_proxy_injects_reasoning_and_strips_streaming_response`, after `tmpdir_path = Path(tmpdir)`, add:

```python
            log_dir = tmpdir_path / "logs"
            log_file = log_dir / "requests.jsonl"
```

In the generated config callback list, add `custom_callbacks.jsonl_logger`:

```yaml
                        - custom_callbacks.request_modifier
                        - custom_callbacks.reasoning_stripper
                        - custom_callbacks.jsonl_logger
```

After constructing `env`, add:

```python
            env["PROXY_LOG_DIR"] = str(log_dir)
```

- [ ] **Step 3: Add a tool-loop request to the existing e2e test**

After the existing streaming request completes, add:

```python
            tool_loop_req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "*",
                        "messages": [
                            {
                                "role": "user",
                                "content": "<system-reminder>Old instruction</system-reminder>\n\nUse the tool output.",
                            },
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "call_1",
                                "content": "tool output",
                            },
                        ],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(tool_loop_req, timeout=15) as resp:
                tool_loop_body = json.loads(resp.read().decode("utf-8"))
```

- [ ] **Step 4: Assert upstream request shape and log audit fields**

Replace:

```python
            assert len(_RecordingHandler.requests) >= 2, "mock upstream did not receive both proxy requests"
```

with:

```python
            assert len(_RecordingHandler.requests) >= 3, "mock upstream did not receive all proxy requests"
```

After the existing stream assertions, add:

```python
            tool_loop_upstream_payload = _RecordingHandler.requests[-1]
            tool_loop_messages = tool_loop_upstream_payload["messages"]
            tool_loop_tail = tool_loop_messages[-1]

            assert tool_loop_messages[0]["content"] == "Use the tool output."
            assert tool_loop_messages[-2]["role"] == "tool"
            assert tool_loop_messages[-2]["content"] == "tool output"
            assert tool_loop_tail["role"] == "user"
            assert isinstance(tool_loop_tail["content"], str)
            assert tool_loop_tail["content"].startswith("<system-reminder>")
            assert "You MUST begin every response" in tool_loop_tail["content"]
            assert tool_loop_tail["content"].rstrip().endswith("</system-reminder>")
            assert "tool output" not in tool_loop_tail["content"]

            assert tool_loop_body["choices"][0]["message"]["content"] == "final visible answer"

            log_entries = _wait_for_jsonl_entries(log_file, 3)
            tool_loop_log = next(
                entry
                for entry in reversed(log_entries)
                if any(
                    message.get("role") == "tool"
                    for message in entry.get("proxy_modified_messages", [])
                )
            )
            assert tool_loop_log["proxy_reasoning_observed"] is True
            assert tool_loop_log["proxy_reasoning_response_mode"] == "non_stream"
            assert "<reasoning>private notes</reasoning>" in tool_loop_log[
                "proxy_raw_response_preview"
            ]
            assert tool_loop_log["proxy_reasoning_reminder_tail_role"] == "user"
            assert tool_loop_log["proxy_reasoning_reminder_tail_injected"] is True
            assert tool_loop_log["proxy_modified_messages"][-1]["role"] == "user"
            assert tool_loop_log["proxy_modified_messages"][-1]["content"].startswith(
                "<system-reminder>"
            )
```

- [ ] **Step 5: Run the e2e test**

Run:

```bash
pytest tests/test_reasoning_proxy_e2e.py -v
```

Expected: PASS. If LiteLLM CLI is unavailable, pytest reports the existing skip reason.

- [ ] **Step 6: Commit e2e coverage**

Run:

```bash
git add tests/test_reasoning_proxy_e2e.py
git commit -m "test: cover reasoning reminder after tool calls"
```

## Task 6: README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update component summary**

Replace the first `RequestModifier` bullet with:

```markdown
1. **`RequestModifier`** — `async_pre_call_hook`, добавляет system-промт, при необходимости дополняет последнее user-сообщение и внедряет reasoning-instruction. Если запрос заканчивается не `user`, добавляет синтетическое trailing `user`-сообщение только с `<system-reminder>...</system-reminder>`.
```

- [ ] **Step 2: Update Reasoning-стриппер section**

Replace the `REASONING_INSTRUCTION` bullet and the following injection paragraph with:

```markdown
- **`REASONING_INSTRUCTION`** — текст reasoning-протокола, который `RequestModifier` добавляет в активный конец запроса. Если последнее сообщение имеет роль `user`, reminder встраивается в него. Если последнее сообщение имеет роль `assistant`, `tool` или другую роль, proxy добавляет в конец синтетическое `user`-сообщение только с reminder. Перед этим старые `<system-reminder>...</system-reminder>` блоки удаляются из исторических user-сообщений, чтобы agent-loop не накапливал инструкции.

Reasoning-instruction впрыскивается для `call_type` из `completion`, `text_completion`, `acompletion`. Это покрывает живой proxy path LiteLLM, где chat completion обычно проходит как `acompletion`.
```

- [ ] **Step 3: Document audit fields**

After the paragraph about `logs/requests.jsonl`, add:

```markdown
Для проверки agent-сценариев лог также содержит proxy audit-поля:

- **`proxy_modified_messages`** — сообщения после модификации proxy, включая trailing reminder.
- **`proxy_reasoning_reminder_tail_role`** и **`proxy_reasoning_reminder_tail_injected`** — проверка, что активный хвост запроса содержит reminder.
- **`proxy_reasoning_observed`**, **`proxy_reasoning_response_mode`**, **`proxy_raw_response_preview`** — bounded evidence, что upstream-ответ содержал `<reasoning>` до очистки.
```

- [ ] **Step 4: Run documentation grep checks**

Run:

```bash
rg -n "synthetic|синтет|proxy_reasoning|trailing|tool" README.md
```

Expected: output includes the new trailing-reminder and audit-field documentation.

- [ ] **Step 5: Commit README changes**

Run:

```bash
git add README.md
git commit -m "docs: document trailing reasoning reminder"
```

## Task 7: Full Automated Verification

**Files:**
- No file edits.

- [ ] **Step 1: Run full pytest suite**

Run:

```bash
pytest -v
```

Expected: all tests pass or the live proxy e2e test is skipped only if `litellm` CLI is unavailable.

- [ ] **Step 2: Check git state**

Run:

```bash
git status --short
```

Expected: only pre-existing unrelated untracked files remain. At the time this plan was written, `docs/superpowers/plans/2026-05-14-reasoning-stripper-user-message.md` was already untracked and must not be committed unless the user asks.

## Task 8: Real Qwen/OpenRouter Coding-Agent Smoke Test

**Files:**
- Create temporary files under `/tmp`; no repo file edits.

- [ ] **Step 1: Confirm required commands and secret are available**

Run:

```bash
test -x .venv/bin/litellm
qwen --version
test -n "$OPENROUTER_API_KEY"
```

Expected: all commands exit `0`. If `OPENROUTER_API_KEY` is empty, set it in the shell from the secret provided by the user, but do not write it to repo files or commit it.

- [ ] **Step 2: Create a temporary OpenRouter proxy config**

Run:

```bash
tmpdir="$(mktemp -d)"
cat > "$tmpdir/config.yaml" <<'YAML'
model_list:
  - model_name: "*"
    litellm_params:
      model: openrouter/qwen/qwen3-vl-30b-a3b-instruct
      api_key: os.environ/OPENROUTER_API_KEY

litellm_settings:
  callbacks:
    - custom_callbacks.request_modifier
    - custom_callbacks.reasoning_stripper
    - custom_callbacks.jsonl_logger
  drop_params: true
YAML
printf '%s\n' "$tmpdir"
```

Expected: prints the temporary directory path.

- [ ] **Step 3: Start the proxy against OpenRouter**

Run:

```bash
export PROXY_LOG_DIR="$tmpdir/logs"
.venv/bin/litellm --config "$tmpdir/config.yaml" --port 4010
```

Expected: the proxy starts and `/health/liveliness` responds on `http://127.0.0.1:4010`. Keep this process running for the next steps.

- [ ] **Step 4: Run Qwen Code through the proxy with tool usage**

In a second shell, run:

```bash
OPENAI_API_KEY=sk-local \
qwen \
  --auth-type openai \
  --openai-base-url http://127.0.0.1:4010/v1 \
  --openai-api-key sk-local \
  --model any-proxy-model \
  --approval-mode yolo \
  --output-format text \
  -p "Use a shell command to print the current working directory, then answer with exactly one short sentence that starts with DONE."
```

Expected: Qwen Code runs at least one tool command and completes with a visible answer that does not include `<reasoning>`.

- [ ] **Step 5: Inspect proxy logs for reasoning evidence**

Run:

```bash
tail -20 "$PROXY_LOG_DIR/requests.jsonl" \
  | rg '"proxy_reasoning_observed": true|<reasoning>|proxy_modified_messages|proxy_reasoning_reminder_tail_injected'
```

Expected: output includes `proxy_reasoning_observed": true`, `proxy_reasoning_reminder_tail_injected": true`, `proxy_modified_messages`, and a bounded raw response preview containing `<reasoning>`.

- [ ] **Step 6: Stop the temporary proxy**

Terminate the `litellm` process from Step 3 with `Ctrl-C`.

Expected: the process exits cleanly. Keep the temporary log directory until the final report is written so the verification evidence can be summarized.
