# Reasoning Injection For `acompletion` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make reasoning-instruction injection work in the live LiteLLM proxy path, including `call_type="acompletion"`, while preserving the existing non-streaming and streaming reasoning stripping.

**Architecture:** Keep `RequestModifier` as the only request-mutation callback and move reasoning injection there. Keep `ReasoningStripper` as a pure response sanitizer for non-streaming and streaming outputs. For string user content, inject the `<system-reminder>...</system-reminder>` block into the string directly instead of converting it to a content array; keep content-array mutation only for messages that already use block arrays.

**Tech Stack:** Python 3.12, LiteLLM proxy callbacks, pytest, pytest-asyncio, stdlib subprocess/http.server/socket/json

---

## Scope Check

This plan covers one subsystem: reasoning request/response handling inside the LiteLLM proxy callback layer. It does not add per-model prompt policies, per-client opt-outs, or persistence of stripped reasoning.

## File Structure

**Modified files:**

- `custom_callbacks.py` - owns request mutation helpers, `RequestModifier`, `ReasoningStripper`, and the callback registration objects.
- `tests/test_custom_callbacks.py` - unit coverage for request mutation, `acompletion` targeting, and response stripping behavior.
- `README.md` - updates ownership of reasoning injection and documents the string-vs-array request behavior.

**Created files:**

- `tests/test_reasoning_proxy_e2e.py` - live proxy regression test that launches a mock upstream and a LiteLLM proxy, then proves reasoning injection and stripping work end to end.

The runtime logic stays in `custom_callbacks.py` to match the current repo shape. The new e2e test gets its own file because it has different setup and failure modes than the lightweight unit tests in `tests/test_custom_callbacks.py`.

## Task 1: Reframe Unit Tests Around `RequestModifier`

**Files:**
- Modify: `tests/test_custom_callbacks.py`
- Modify: `custom_callbacks.py`

- [ ] **Step 1: Write failing unit tests for request-side reasoning injection in `RequestModifier`**

At the top of `tests/test_custom_callbacks.py`, replace the current import block:

```python
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_callbacks import (
    ReasoningStreamFilter,
    ReasoningStripper,
    REASONING_INSTRUCTION,
)
```

with:

```python
import pytest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_callbacks import (
    REASONING_INSTRUCTION,
    ReasoningStreamFilter,
    ReasoningStripper,
    RequestModifier,
)
```

Then replace the current `TestReasoningStripperPreHook` class with this class:

```python
class TestRequestModifierReasoningInjection:
    def setup_method(self):
        self.modifier = RequestModifier()

    @pytest.mark.asyncio
    async def test_injects_reasoning_reminder_into_string_user_message_for_acompletion(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        }

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], str)
        assert last_msg["content"].startswith("<system-reminder>")
        assert "You MUST begin every response" in last_msg["content"]
        assert last_msg["content"].endswith("What is 2+2?")

    @pytest.mark.asyncio
    async def test_injects_reasoning_reminder_into_content_array_for_acompletion(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ]
        }

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        content = result["messages"][-1]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert "<system-reminder>" in content[0]["text"]
        assert content[1]["text"] == "Analyze this:"
        assert content[2]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_creates_string_user_message_when_none_exist_for_acompletion(self):
        data = {"messages": [{"role": "system", "content": "You are helpful."}]}

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], str)
        assert last_msg["content"].startswith("<system-reminder>")

    @pytest.mark.asyncio
    async def test_removes_existing_system_reminder_before_reinserting_for_acompletion(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "<SYSTEM-REMINDER>Old instruction</SYSTEM-REMINDER>\n\nUser query",
                }
            ]
        }

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        content = result["messages"][-1]["content"]
        assert content.count("<system-reminder>") == 1
        assert content.endswith("User query")

    @pytest.mark.asyncio
    async def test_skips_reasoning_injection_for_non_target_call_types(self):
        data = {"messages": [{"role": "user", "content": "test"}]}

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="embeddings",
        )

        assert result["messages"][0]["content"] == "test"
```

Add one more regression test after that class:

```python
class TestReasoningStripperRequestPath:
    def setup_method(self):
        self.stripper = ReasoningStripper()

    @pytest.mark.asyncio
    async def test_pre_call_hook_is_noop_for_acompletion(self):
        data = {"messages": [{"role": "user", "content": "test"}]}

        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        assert result == data
```

- [ ] **Step 2: Run the request-path unit tests and verify they fail**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py -k 'RequestModifierReasoningInjection or ReasoningStripperRequestPath' -v
```

Expected: FAIL. The string-content tests should fail because `RequestModifier` currently does not inject `REASONING_INSTRUCTION`, and the noop test should fail because `ReasoningStripper` still mutates request data.

- [ ] **Step 3: Implement shared request-mutation helpers and move reasoning injection into `RequestModifier`**

In `custom_callbacks.py`, add the helper set near the existing reminder regex definitions:

```python
_TARGET_REASONING_CALL_TYPES = {"completion", "text_completion", "acompletion"}


def _is_reasoning_target_call(call_type: str) -> bool:
    return call_type in _TARGET_REASONING_CALL_TYPES


def _clean_system_reminder_text(text: str) -> str:
    return _SYSTEM_REMINDER_RE.sub("", text).strip()


def _inject_reasoning_reminder_into_string(content: str) -> str:
    cleaned = _clean_system_reminder_text(content)
    if cleaned:
        return _SYSTEM_REMINDER_BLOCK_TEXT + cleaned
    return _SYSTEM_REMINDER_BLOCK_TEXT.rstrip()


def _inject_reasoning_reminder_into_blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered_content = []
    for block in content:
        if not isinstance(block, dict):
            filtered_content.append(block)
            continue
        if block.get("type") != "text":
            filtered_content.append(block)
            continue

        text = block.get("text", "")
        cleaned = _clean_system_reminder_text(text)
        if cleaned:
            filtered_content.append({"type": "text", "text": cleaned})

    return [_INSTRUCTION_BLOCK] + filtered_content
```

Then update `RequestModifier.async_pre_call_hook` so the body becomes:

```python
        if call_type not in ("completion", "text_completion", "acompletion"):
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

        if REASONING_INSTRUCTION and _is_reasoning_target_call(call_type):
            last_user_idx = next(
                (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
                None,
            )

            if last_user_idx is None:
                messages.append(
                    {
                        "role": "user",
                        "content": _SYSTEM_REMINDER_BLOCK_TEXT.rstrip(),
                    }
                )
            else:
                user_msg = messages[last_user_idx]
                content = user_msg.get("content")

                if isinstance(content, str):
                    user_msg["content"] = _inject_reasoning_reminder_into_string(content)
                elif isinstance(content, list):
                    user_msg["content"] = _inject_reasoning_reminder_into_blocks(content)
                else:
                    user_msg["content"] = _SYSTEM_REMINDER_BLOCK_TEXT.rstrip()

        data["messages"] = messages
        return data
```

Finally, replace `ReasoningStripper.async_pre_call_hook` with a noop implementation:

```python
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "acompletion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[dict]:
        return data
```

- [ ] **Step 4: Run the request-path unit tests and verify they pass**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py -k 'RequestModifierReasoningInjection or ReasoningStripperRequestPath' -v
```

Expected: PASS, with 6 tests passing.

- [ ] **Step 5: Commit the request-path refactor**

Run:

```bash
git add custom_callbacks.py tests/test_custom_callbacks.py
git commit -m "fix: move reasoning injection into request modifier"
```

## Task 2: Align Existing Callback Tests With The New Ownership Boundary

**Files:**
- Modify: `tests/test_custom_callbacks.py`
- Modify: `custom_callbacks.py`

- [ ] **Step 1: Write failing unit tests for `acompletion` coverage and string-preserving behavior**

In `tests/test_custom_callbacks.py`, add these methods inside `TestRequestModifierReasoningInjection`:

```python
    @pytest.mark.asyncio
    async def test_string_user_content_stays_string_after_injection(self):
        data = {"messages": [{"role": "user", "content": "hello"}]}

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        assert isinstance(result["messages"][0]["content"], str)
        assert result["messages"][0]["content"].endswith("hello")

    @pytest.mark.asyncio
    async def test_user_prompt_append_still_runs_before_reasoning_injection(self, monkeypatch):
        monkeypatch.setattr("custom_callbacks.USER_PROMPT_APPEND", "\n\n[EXTRA]")
        data = {"messages": [{"role": "user", "content": "hello"}]}

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        content = result["messages"][0]["content"]
        assert content.startswith("<system-reminder>")
        assert content.endswith("hello\n\n[EXTRA]")
```

Also update the existing non-target call-type test expectations anywhere they still assume only `"completion"` / `"text_completion"` are supported. Every request-side reasoning test should use `"acompletion"` as the primary success path.

- [ ] **Step 2: Run the focused unit tests and verify at least one fails**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py -k 'string_user_content_stays_string or user_prompt_append_still_runs_before_reasoning_injection' -v
```

Expected: FAIL if the helper ordering or string-preservation logic is off.

- [ ] **Step 3: Minimal implementation cleanup**

If the Step 2 failures expose ordering problems, keep the implementation minimal:

- keep `USER_PROMPT_APPEND` before reasoning injection so the reminder remains first;
- keep string request content as `str`;
- do not reintroduce conversion from `str` to `list`.

The final request-side helper section in `custom_callbacks.py` should still include these exact contracts:

```python
def _inject_reasoning_reminder_into_string(content: str) -> str:
    cleaned = _clean_system_reminder_text(content)
    if cleaned:
        return _SYSTEM_REMINDER_BLOCK_TEXT + cleaned
    return _SYSTEM_REMINDER_BLOCK_TEXT.rstrip()


def _is_reasoning_target_call(call_type: str) -> bool:
    return call_type in {"completion", "text_completion", "acompletion"}
```

- [ ] **Step 4: Run the full unit suite and verify it passes**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py -v
```

Expected: PASS for the full callback unit suite.

- [ ] **Step 5: Commit the unit-test alignment**

Run:

```bash
git add custom_callbacks.py tests/test_custom_callbacks.py
git commit -m "test: cover reasoning injection for acompletion"
```

## Task 3: Add A Live Proxy Regression Test

**Files:**
- Create: `tests/test_reasoning_proxy_e2e.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing end-to-end regression test**

Create `tests/test_reasoning_proxy_e2e.py` with this content:

```python
import json
import os
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _RecordingHandler(BaseHTTPRequestHandler):
    requests = []
    stream_events = [
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": "mock-model", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "<rea"}, "finish_reason": None}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": "mock-model", "choices": [{"index": 0, "delta": {"content": "soning>private notes"}, "finish_reason": None}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": "mock-model", "choices": [{"index": 0, "delta": {"content": "</rea"}, "finish_reason": None}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": "mock-model", "choices": [{"index": 0, "delta": {"content": "soning>final"}, "finish_reason": None}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": "mock-model", "choices": [{"index": 0, "delta": {"content": " visible answer"}, "finish_reason": None}]},
        {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": "mock-model", "choices": [{"index": 0, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
    ]

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) if length else b"{}")
        type(self).requests.append(payload)

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in type(self).stream_events:
                self.wfile.write(f"data: {json.dumps(event)}\\n\\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"data: [DONE]\\n\\n")
            self.wfile.flush()
            self.close_connection = True
            return

        body = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<reasoning>private notes</reasoning>final visible answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


@pytest.mark.integration
def test_proxy_injects_reasoning_and_strips_streaming_response():
    repo_root = Path(__file__).resolve().parents[1]
    upstream_port = _free_port()
    proxy_port = _free_port()

    server = ThreadingHTTPServer(("127.0.0.1", upstream_port), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        config_path = tmpdir_path / "config.yaml"
        callback_link = tmpdir_path / "custom_callbacks.py"
        callback_link.symlink_to(repo_root / "custom_callbacks.py")

        config_path.write_text(
            textwrap.dedent(
                f\"\"\"\
                model_list:
                  - model_name: "*"
                    litellm_params:
                      model: hosted_vllm/MODEL_PLACEHOLDER
                      api_base: http://127.0.0.1:{upstream_port}/v1

                litellm_settings:
                  callbacks:
                    - custom_callbacks.request_modifier
                    - custom_callbacks.reasoning_stripper
                \"\"\"
            ),
            encoding="utf-8",
        )

        proc = subprocess.Popen(
            [str(repo_root / ".venv/bin/litellm"), "--config", str(config_path), "--port", str(proxy_port)],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            deadline = time.time() + 20
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/health/liveliness", timeout=1):
                        break
                except Exception:
                    time.sleep(0.25)
            else:
                output = proc.stdout.read() if proc.stdout else ""
                raise AssertionError(f"proxy did not start\\n{output}")

            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/chat/completions",
                data=json.dumps(
                    {
                        "model": "hosted_vllm/MODEL_PLACEHOLDER",
                        "messages": [{"role": "user", "content": "Return the visible answer only."}],
                        "stream": True,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8")

            upstream_payload = _RecordingHandler.requests[-1]
            user_content = upstream_payload["messages"][-1]["content"]
            assert isinstance(user_content, str)
            assert user_content.startswith("<system-reminder>")
            assert "You MUST begin every response" in user_content
            assert user_content.endswith("Return the visible answer only.")
            assert "<reasoning>" not in body
            assert "final visible answer" in body
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            server.shutdown()
            server.server_close()
```

- [ ] **Step 2: Run the e2e regression test and verify it fails**

Run:

```bash
.venv/bin/pytest tests/test_reasoning_proxy_e2e.py::test_proxy_injects_reasoning_and_strips_streaming_response -v
```

Expected: FAIL on the upstream payload assertion because the live proxy currently forwards the original string user content without the injected `<system-reminder>` block.

- [ ] **Step 3: Make the e2e test pass**

No new production module is required here. Use the implementation from Tasks 1-2. If the test still fails after those changes, only make minimal integration fixes:

- include `"acompletion"` in the target call-type helper;
- keep the injected string payload as `str`;
- keep `ReasoningStripper` registered for response cleanup;
- do not add extra callback objects or reorder `callbacks` in `config.yaml`.

The live path is correct only when these assertions hold:

```python
assert isinstance(user_content, str)
assert user_content.startswith("<system-reminder>")
assert "You MUST begin every response" in user_content
assert user_content.endswith("Return the visible answer only.")
assert "<reasoning>" not in body
assert "final visible answer" in body
```

- [ ] **Step 4: Run the e2e regression test again and verify it passes**

Run:

```bash
.venv/bin/pytest tests/test_reasoning_proxy_e2e.py::test_proxy_injects_reasoning_and_strips_streaming_response -v
```

Expected: PASS.

- [ ] **Step 5: Commit the e2e regression coverage**

Run:

```bash
git add tests/test_reasoning_proxy_e2e.py custom_callbacks.py
git commit -m "test: add live proxy reasoning injection regression"
```

## Task 4: Update Docs And Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `custom_callbacks.py`
- Modify: `tests/test_custom_callbacks.py`
- Modify: `tests/test_reasoning_proxy_e2e.py`

- [ ] **Step 1: Update README so it matches the final behavior**

In `README.md`, change the component summary from:

```markdown
2. **`ReasoningStripper`** — `async_pre_call_hook` + `async_post_call_success_hook` + `async_post_call_streaming_iterator_hook`: просит модель оборачивать reasoning в XML-тег и вырезает этот блок из non-streaming и streaming-ответов перед отдачей клиенту.
```

to:

```markdown
2. **`ReasoningStripper`** — `async_post_call_success_hook` + `async_post_call_streaming_iterator_hook`: вырезает `<reasoning>...</reasoning>` из non-streaming и streaming-ответов перед отдачей клиенту.
```

Then replace the current Reasoning-stripper request-injection description with:

```markdown
- **`REASONING_INSTRUCTION`** — текст reasoning-протокола, который `RequestModifier` добавляет в последнее `user`-сообщение. Для строкового `content` reminder встраивается в строку напрямую. Для `content`-массивов callback добавляет первый text-блок с reminder и сохраняет остальные блоки.

Reasoning-instruction впрыскивается для `call_type` из `completion`, `text_completion`, `acompletion`. Это покрывает живой proxy path LiteLLM, где chat completion обычно проходит как `acompletion`.
```

- [ ] **Step 2: Run the complete verification set**

Run:

```bash
.venv/bin/pytest -v
```

Expected: PASS for the full suite, including `tests/test_custom_callbacks.py` and `tests/test_reasoning_proxy_e2e.py`.

Then run:

```bash
python3 - <<'PY'
import ast
from pathlib import Path

source = Path("custom_callbacks.py").read_text(encoding="utf-8")
module = ast.parse(source)
exports = {
    node.targets[0].id
    for node in module.body
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
}
assert "request_modifier" in exports
assert "reasoning_stripper" in exports
print("reasoning callback import check passed")
PY
```

Expected:

```text
reasoning callback import check passed
```

- [ ] **Step 3: Review the worktree before the final commit**

Run:

```bash
git status --short
```

Expected: only these files should be staged or modified for this work:

```text
M README.md
M custom_callbacks.py
M tests/test_custom_callbacks.py
A tests/test_reasoning_proxy_e2e.py
```

Leave the unrelated untracked file `docs/superpowers/plans/2026-05-14-reasoning-stripper-user-message.md` alone.

- [ ] **Step 4: Commit the finished implementation**

Run:

```bash
git add README.md custom_callbacks.py tests/test_custom_callbacks.py tests/test_reasoning_proxy_e2e.py
git commit -m "fix: inject reasoning instruction for acompletion"
```

- [ ] **Step 5: Record the final smoke-test commands in the commit message or handoff notes**

Use this exact verification summary in the handoff:

```text
.venv/bin/pytest -v
.venv/bin/pytest tests/test_reasoning_proxy_e2e.py::test_proxy_injects_reasoning_and_strips_streaming_response -v
python3 - <<'PY' ... reasoning callback import check passed
```

## Self-Review

Spec coverage:
- Request-side reasoning injection moved to `RequestModifier`: covered by Tasks 1 and 2.
- `acompletion` support: covered by Tasks 1, 2, and 3.
- String content must stay string: covered by Tasks 1, 2, and 3.
- Response-side stripping must continue to work: covered by Task 3 and full-suite verification in Task 4.
- Docs must reflect the ownership split: covered by Task 4.

Placeholder scan:
- No `TODO`, `TBD`, or “similar to previous task” placeholders remain.

Type consistency:
- `RequestModifier` owns request mutation in every task.
- `ReasoningStripper` is treated as request-path noop and response-path sanitizer in every task.
- `call_type` targeting consistently uses `completion`, `text_completion`, and `acompletion`.
