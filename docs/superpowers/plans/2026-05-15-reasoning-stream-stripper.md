# Reasoning Stream Stripper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add robust streaming removal of hidden `<reasoning>...</reasoning>` blocks while keeping the existing last-user-message instruction injection.

**Architecture:** Keep the LiteLLM callback in `custom_callbacks.py`. Add a small `ReasoningStreamFilter` state machine for chunk-safe tag stripping, wire it into `ReasoningStripper.async_post_call_streaming_iterator_hook`, and reuse a stricter text cleanup helper for non-streaming responses. Tests stay in `tests/test_custom_callbacks.py` and use lightweight `SimpleNamespace` chunks to exercise the callback without a real LiteLLM server.

**Tech Stack:** Python 3.12, LiteLLM proxy callbacks, pytest, pytest-asyncio

---

## Scope Check

This plan covers one subsystem: the `ReasoningStripper` callback. It does not add per-client config, model routing, persistent reasoning logs, or non-chat endpoint support.

## File Structure

**Modified files:**

- `custom_callbacks.py` - owns request injection, non-streaming response cleanup, streaming iterator cleanup, and the new stream filter helper.
- `tests/test_custom_callbacks.py` - extends existing unit tests with stream filter and streaming hook coverage.
- `README.md` - updates the ReasoningStripper section so documented behavior matches streaming cleanup.

No new runtime module is needed. The helper is small and private to the callback file, so splitting it into another file would add import overhead without improving the boundaries.

## Task 1: Add The Streaming Text Filter

**Files:**
- Modify: `tests/test_custom_callbacks.py`
- Modify: `custom_callbacks.py`

- [ ] **Step 1: Write failing unit tests for `ReasoningStreamFilter`**

In `tests/test_custom_callbacks.py`, replace the current imports at the top:

```python
import pytest
from unittest.mock import MagicMock
from custom_callbacks import ReasoningStripper, REASONING_INSTRUCTION
```

with:

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

Then append this test class after the existing `TestReasoningStripperPreHook` class:

```python
class TestReasoningStreamFilter:
    def test_removes_reasoning_when_tags_are_in_one_chunk(self):
        stream_filter = ReasoningStreamFilter("reasoning")

        visible = stream_filter.feed(
            "<reasoning>private notes</reasoning>\n\nThe answer is 42."
        )
        visible += stream_filter.flush()

        assert visible == "The answer is 42."

    def test_removes_reasoning_when_tags_are_split_across_chunks(self):
        stream_filter = ReasoningStreamFilter("reasoning")
        visible_parts = []

        for fragment in [
            "<rea",
            "soning>private",
            " notes</rea",
            "soning>\n\nThe answer",
            " is 42.",
        ]:
            visible_parts.append(stream_filter.feed(fragment))
        visible_parts.append(stream_filter.flush())

        assert "".join(visible_parts) == "The answer is 42."

    def test_passes_through_text_when_no_reasoning_tag_exists(self):
        stream_filter = ReasoningStreamFilter("reasoning")

        visible = stream_filter.feed("Hello ")
        visible += stream_filter.feed("world")
        visible += stream_filter.flush()

        assert visible == "Hello world"

    def test_drops_unclosed_reasoning_block_on_flush(self):
        stream_filter = ReasoningStreamFilter("reasoning")

        visible = stream_filter.feed("<reasoning>private notes")
        visible += stream_filter.feed(" still private")
        visible += stream_filter.flush()

        assert visible == ""

    def test_flush_releases_held_partial_tag_outside_reasoning(self):
        stream_filter = ReasoningStreamFilter("reasoning")

        visible = stream_filter.feed("Literal <rea")
        visible += stream_filter.flush()

        assert visible == "Literal <rea"
```

- [ ] **Step 2: Run the new filter tests and verify they fail**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py::TestReasoningStreamFilter -v
```

Expected: FAIL during collection with an import error like:

```text
ImportError: cannot import name 'ReasoningStreamFilter' from 'custom_callbacks'
```

- [ ] **Step 3: Add the stream filter implementation**

In `custom_callbacks.py`, change the typing import:

```python
from typing import Literal, Optional
```

to:

```python
from typing import Any, AsyncGenerator, Literal, Optional
```

Then insert this code immediately after `_xml_tag_re`:

```python
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
                self._drop_leading_ws = True
                continue

            if self._drop_leading_ws:
                self._buffer = self._buffer.lstrip()
                self._drop_leading_ws = False
                if not self._buffer:
                    break

            opening_match = self._opening_re.search(self._buffer)
            if opening_match is None:
                visible, self._buffer = self._outside_visible_prefix()
                visible_parts.append(visible)
                break

            visible_parts.append(self._buffer[:opening_match.start()])
            self._buffer = self._buffer[opening_match.end():]
            self._inside_reasoning = True

        return "".join(visible_parts)

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
```

While editing this file, remove this stray comment if it is still present:

```python
# ASINC !!!!
```

- [ ] **Step 4: Run the filter tests and verify they pass**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py::TestReasoningStreamFilter -v
```

Expected: PASS, with 5 tests passing.

- [ ] **Step 5: Commit the filter**

Run:

```bash
git add custom_callbacks.py tests/test_custom_callbacks.py
git commit -m "feat: add reasoning stream text filter"
```

Expected: commit succeeds and only `custom_callbacks.py` plus `tests/test_custom_callbacks.py` are staged.

## Task 2: Wire The Filter Into The Streaming Iterator Hook

**Files:**
- Modify: `tests/test_custom_callbacks.py`
- Modify: `custom_callbacks.py`

- [ ] **Step 1: Add streaming hook test helpers**

In `tests/test_custom_callbacks.py`, insert these helpers after the imports and before `class TestReasoningStripperPreHook`:

```python
def _stream_chunk(content=None, finish_reason=None, role=None, usage=None):
    delta = SimpleNamespace()
    if content is not None:
        delta.content = content
    if role is not None:
        delta.role = role

    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    chunk = SimpleNamespace(choices=[choice])
    if usage is not None:
        chunk.usage = usage
    return chunk


async def _collect_streaming_hook(stripper, chunks):
    async def source():
        for chunk in chunks:
            yield chunk

    return [
        chunk
        async for chunk in stripper.async_post_call_streaming_iterator_hook(
            user_api_key_dict=None,
            response=source(),
            request_data={},
        )
    ]


def _chunk_content(chunk):
    return getattr(chunk.choices[0].delta, "content", None)
```

- [ ] **Step 2: Add failing streaming hook tests**

Append this test class after `TestReasoningStreamFilter`:

```python
class TestReasoningStripperStreamingHook:
    def setup_method(self):
        self.stripper = ReasoningStripper()

    @pytest.mark.asyncio
    async def test_streaming_hook_strips_reasoning_split_across_chunks(self):
        chunks = [
            _stream_chunk("<rea", role="assistant"),
            _stream_chunk("soning>private notes"),
            _stream_chunk("</rea"),
            _stream_chunk("soning>\n\nThe answer"),
            _stream_chunk(" is 42."),
            _stream_chunk(finish_reason="stop"),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_chunk_content(chunk) or "" for chunk in result) == (
            "The answer is 42."
        )
        assert result[0].choices[0].delta.role == "assistant"
        assert result[-1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_streaming_hook_passes_through_text_without_reasoning(self):
        chunks = [
            _stream_chunk("Hello ", role="assistant"),
            _stream_chunk("world"),
            _stream_chunk(finish_reason="stop"),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_chunk_content(chunk) or "" for chunk in result) == "Hello world"
        assert result[0].choices[0].delta.role == "assistant"
        assert result[-1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_streaming_hook_preserves_finish_and_usage_chunk(self):
        usage = {"completion_tokens": 8, "prompt_tokens": 4, "total_tokens": 12}
        chunks = [
            _stream_chunk("<reasoning>private</reasoning>\n\nVisible"),
            _stream_chunk(finish_reason="stop", usage=usage),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_chunk_content(chunk) or "" for chunk in result) == "Visible"
        assert result[-1].choices[0].finish_reason == "stop"
        assert result[-1].usage == usage

    @pytest.mark.asyncio
    async def test_streaming_hook_flushes_held_visible_tail_on_finish(self):
        chunks = [
            _stream_chunk("Literal <rea", role="assistant"),
            _stream_chunk(finish_reason="stop"),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_chunk_content(chunk) or "" for chunk in result) == "Literal <rea"
        assert result[-1].choices[0].finish_reason == "stop"
```

- [ ] **Step 3: Run streaming hook tests and verify they fail**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py::TestReasoningStripperStreamingHook -v
```

Expected: FAIL with an assertion error because the inherited base hook passes chunks through unchanged, so the joined visible text still contains raw reasoning content:

```text
AssertionError
```

- [ ] **Step 4: Add streaming chunk helpers**

In `custom_callbacks.py`, insert these helpers after `_INSTRUCTION_BLOCK`:

```python
def _stream_choice_index(choice: Any, fallback: int) -> int:
    index = getattr(choice, "index", fallback)
    if isinstance(index, int):
        return index
    return fallback


def _get_delta_content(choice: Any) -> Optional[str]:
    delta = getattr(choice, "delta", None)
    if isinstance(delta, dict):
        content = delta.get("content")
    else:
        content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    return None


def _set_delta_content(choice: Any, content: str) -> None:
    delta = getattr(choice, "delta", None)
    if delta is None:
        return
    if isinstance(delta, dict):
        delta["content"] = content
        return
    setattr(delta, "content", content)
```

- [ ] **Step 5: Add `async_post_call_streaming_iterator_hook`**

In `custom_callbacks.py`, add this method inside `class ReasoningStripper`, directly after `async_post_call_success_hook`:

```python
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        filters: dict[int, ReasoningStreamFilter] = {}

        async for chunk in response:
            try:
                choices = getattr(chunk, "choices", None) or []
                for position, choice in enumerate(choices):
                    choice_index = _stream_choice_index(choice, position)
                    stream_filter = filters.setdefault(
                        choice_index,
                        ReasoningStreamFilter(REASONING_TAG),
                    )

                    content = _get_delta_content(choice)
                    if content is not None:
                        _set_delta_content(choice, stream_filter.feed(content))

                    if getattr(choice, "finish_reason", None) is not None:
                        tail = stream_filter.flush()
                        if tail:
                            current = _get_delta_content(choice) or ""
                            _set_delta_content(choice, current + tail)
                        filters.pop(choice_index, None)
            except Exception:
                yield chunk
                continue

            yield chunk
```

- [ ] **Step 6: Run streaming hook tests and verify they pass**

Run:

```bash
.venv/bin/pytest tests/test_custom_callbacks.py::TestReasoningStripperStreamingHook -v
```

Expected: PASS, with 4 tests passing.

- [ ] **Step 7: Commit the streaming hook**

Run:

```bash
git add custom_callbacks.py tests/test_custom_callbacks.py
git commit -m "feat: strip reasoning from streaming chunks"
```

Expected: commit succeeds and only `custom_callbacks.py` plus `tests/test_custom_callbacks.py` are staged.

## Task 3: Strengthen Non-Streaming Cleanup

**Files:**
- Modify: `tests/test_custom_callbacks.py`
- Modify: `custom_callbacks.py`

- [ ] **Step 1: Add failing non-streaming cleanup tests**

Inside `class TestReasoningStripperPreHook`, after `test_post_hook_strips_reasoning_tags`, add:

```python
    @pytest.mark.asyncio
    async def test_post_hook_drops_unclosed_reasoning_block(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "<reasoning>private notes that never close"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = await self.stripper.async_post_call_success_hook(
            data={},
            user_api_key_dict=None,
            response=mock_response,
        )

        assert result.choices[0].message.content == ""

    @pytest.mark.asyncio
    async def test_post_hook_strips_multiple_reasoning_blocks(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = (
            "<reasoning>first private block</reasoning>\n\n"
            "Visible one. "
            "<reasoning>second private block</reasoning>\n\n"
            "Visible two."
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = await self.stripper.async_post_call_success_hook(
            data={},
            user_api_key_dict=None,
            response=mock_response,
        )

        assert result.choices[0].message.content == "Visible one. Visible two."
```

- [ ] **Step 2: Run the new non-streaming tests and verify failure**

Run:

```bash
.venv/bin/pytest \
  tests/test_custom_callbacks.py::TestReasoningStripperPreHook::test_post_hook_drops_unclosed_reasoning_block \
  tests/test_custom_callbacks.py::TestReasoningStripperPreHook::test_post_hook_strips_multiple_reasoning_blocks \
  -v
```

Expected: `test_post_hook_drops_unclosed_reasoning_block` fails because unclosed reasoning still leaks.

- [ ] **Step 3: Add shared non-streaming cleanup helper**

In `custom_callbacks.py`, insert this code after `_REASONING_RE = _xml_tag_re(REASONING_TAG)`:

```python
_UNCLOSED_REASONING_RE = re.compile(
    rf"<{re.escape(REASONING_TAG)}\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)


def _strip_reasoning_text(content: str) -> str:
    return _UNCLOSED_REASONING_RE.sub("", _REASONING_RE.sub("", content))
```

Then change the post-hook body from:

```python
            if isinstance(content, str):
                message.content = _REASONING_RE.sub("", content)
```

to:

```python
            if isinstance(content, str):
                message.content = _strip_reasoning_text(content)
```

- [ ] **Step 4: Run non-streaming cleanup tests and verify they pass**

Run:

```bash
.venv/bin/pytest \
  tests/test_custom_callbacks.py::TestReasoningStripperPreHook::test_post_hook_strips_reasoning_tags \
  tests/test_custom_callbacks.py::TestReasoningStripperPreHook::test_post_hook_drops_unclosed_reasoning_block \
  tests/test_custom_callbacks.py::TestReasoningStripperPreHook::test_post_hook_strips_multiple_reasoning_blocks \
  -v
```

Expected: PASS, with 3 tests passing.

- [ ] **Step 5: Commit non-streaming cleanup**

Run:

```bash
git add custom_callbacks.py tests/test_custom_callbacks.py
git commit -m "fix: drop unclosed reasoning in non-stream responses"
```

Expected: commit succeeds and only `custom_callbacks.py` plus `tests/test_custom_callbacks.py` are staged.

## Task 4: Update Documentation And Run Full Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_custom_callbacks.py`

- [ ] **Step 1: Update README ReasoningStripper behavior**

In `README.md`, replace the current section beginning with:

```markdown
## Reasoning-стриппер
```

through the paragraph that ends with:

```markdown
Незакрытые блоки (например, из-за обрезания по `max_tokens`) остаются нетронутыми, чтобы клиент мог их увидеть и обработать.
```

with:

```markdown
## Reasoning-стриппер

В `custom_callbacks.py`, секция «Удаление reasoning-тега из ответа модели»:

- **`REASONING_TAG`** — имя XML-тега (по умолчанию `reasoning`).
- **`REASONING_INSTRUCTION`** — текст, добавляемый в последнее `user`-сообщение как первый text-блок внутри `<system-reminder>...</system-reminder>`. Пустая строка выключает добавление инструкции, но очистка ответа продолжит работать.

Перед отправкой запроса callback удаляет старые `<system-reminder>...</system-reminder>` блоки из последнего `user`-сообщения и добавляет свежую инструкцию. Это предотвращает накопление одинаковых reminder-блоков в агентских сценариях.

После ответа модели callback удаляет `<reasoning>...</reasoning>` из non-streaming ответов и из streaming-чанков. Потоковая очистка работает с тегами, разорванными между чанками, и сохраняет служебные чанки с `role`, `tool_calls`, `usage` и `finish_reason`.

Если модель открыла `<reasoning>`, но не закрыла тег, содержимое reasoning не отдаётся клиенту.
```

- [ ] **Step 2: Run the complete unit test suite**

Run:

```bash
.venv/bin/pytest -v
```

Expected: PASS, including the existing pre-hook tests and the new streaming tests.

- [ ] **Step 3: Run a focused import check**

Run:

```bash
.venv/bin/python - <<'PY'
from custom_callbacks import ReasoningStripper, ReasoningStreamFilter

stream_filter = ReasoningStreamFilter("reasoning")
assert stream_filter.feed("<reasoning>hidden</reasoning>\n\nVisible") == "Visible"
assert stream_filter.flush() == ""
assert ReasoningStripper() is not None
print("reasoning callback import check passed")
PY
```

Expected output:

```text
reasoning callback import check passed
```

- [ ] **Step 4: Run a manual streaming smoke test when the configured backend is available**

Start the proxy in one terminal:

```bash
set -a && [ -f .env ] && source .env; set +a
.venv/bin/litellm --config config.yaml --port 4000
```

In another terminal, send a streaming request:

```bash
curl -N http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "any-name-you-like",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say exactly: final visible answer"}
    ]
  }'
```

Expected: the SSE stream reaches `[DONE]`, visible deltas do not include `<reasoning>`, `</reasoning>`, or private reasoning text, and the final visible answer is still present.

- [ ] **Step 5: Commit docs and verification-ready state**

Run:

```bash
git add README.md
git commit -m "docs: document streaming reasoning stripping"
```

Expected: commit succeeds with only `README.md` staged. If Step 2 or Step 3 fails, fix the failing code before this commit.

## Final Verification

Run:

```bash
git status --short
.venv/bin/pytest -v
```

Expected:

```text
pytest exits with status 0
```

`git status --short` may show unrelated pre-existing files. It should not show uncommitted changes in `custom_callbacks.py`, `tests/test_custom_callbacks.py`, or `README.md` after the task commits.
