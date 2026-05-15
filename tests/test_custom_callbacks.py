from copy import deepcopy

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_callbacks import (
    RequestModifier,
    REASONING_INSTRUCTION,
    ReasoningStreamFilter,
    ReasoningStripper,
)


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
    choice = chunk.choices[0]
    if isinstance(choice, dict):
        return choice.get("delta", {}).get("content")
    return getattr(choice.delta, "content", None)


def _choice_content(choice):
    if isinstance(choice, dict):
        return choice.get("delta", {}).get("content")
    return getattr(choice.delta, "content", None)


class _ChoiceWithBrokenFinishReason:
    index = 0

    def __init__(self, content):
        self.delta = SimpleNamespace(content=content)

    @property
    def finish_reason(self):
        raise RuntimeError("finish reason unavailable")


class TestRequestModifierReasoningInjection:
    def setup_method(self):
        self.modifier = RequestModifier()

    @pytest.mark.asyncio
    async def test_injects_reasoning_reminder_into_string_user_message_for_acompletion(
        self,
    ):
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
        assert REASONING_INSTRUCTION in last_msg["content"]
        assert last_msg["content"].endswith("What is 2+2?")

    @pytest.mark.asyncio
    async def test_preserves_string_whitespace_when_injecting_reasoning_reminder(self):
        data = {
            "messages": [
                {"role": "user", "content": "  padded user text  "},
            ]
        }

        result = await self.modifier.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=deepcopy(data),
            call_type="acompletion",
        )

        content = result["messages"][-1]["content"]
        assert isinstance(content, str)
        assert content.endswith("  padded user text  ")

    @pytest.mark.asyncio
    async def test_injects_reasoning_reminder_into_content_array_for_acompletion(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.png"},
                        },
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
        assert REASONING_INSTRUCTION in content[0]["text"]
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Analyze this:"
        assert content[2]["type"] == "image_url"
        assert content[2]["image_url"]["url"] == "https://example.com/img.png"

    @pytest.mark.asyncio
    async def test_creates_string_user_message_when_none_exist_for_acompletion(self):
        data = {
            "messages": [{"role": "system", "content": "You are helpful."}]
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
        assert REASONING_INSTRUCTION in last_msg["content"]

    @pytest.mark.asyncio
    async def test_removes_existing_system_reminder_before_reinserting_for_acompletion(
        self,
    ):
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
        assert isinstance(content, str)
        assert content.count("<system-reminder>") == 1
        assert content.endswith("User query")
        assert "Old instruction" not in content

    @pytest.mark.asyncio
    async def test_removes_existing_system_reminder_from_array_and_preserves_non_text_blocks_for_acompletion(
        self,
    ):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>Old instruction</system-reminder>\n\nFirst part",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.png"},
                        },
                        {"type": "text", "text": "Final query"},
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
        assert len(content) == 4
        assert content[0]["type"] == "text"
        assert "<system-reminder>" in content[0]["text"]
        assert content[1] == {"type": "text", "text": "First part"}
        assert content[2]["type"] == "image_url"
        assert content[2]["image_url"]["url"] == "https://example.com/img.png"
        assert content[3] == {"type": "text", "text": "Final query"}

    @pytest.mark.asyncio
    async def test_preserves_array_text_whitespace_when_cleaning_existing_reminder(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>Old instruction</system-reminder>\n\n  padded block  ",
                        }
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
        assert content[1] == {"type": "text", "text": "  padded block  "}

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

    @pytest.mark.asyncio
    async def test_post_hook_strips_reasoning_tags(self):
        """Post-hook should still strip <reasoning>...</reasoning> tags from response."""
        # Mock response with reasoning tags
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = (
            "<reasoning>Let me think about this...</reasoning>\n\n"
            "The answer is 42."
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = await self.stripper.async_post_call_success_hook(
            data={},
            user_api_key_dict=None,
            response=mock_response
        )

        # Should strip reasoning tags but keep the answer
        assert result.choices[0].message.content == "The answer is 42."
        assert "<reasoning>" not in result.choices[0].message.content

    @pytest.mark.asyncio
    async def test_post_hook_strips_dict_shaped_response(self):
        response = {
            "choices": [
                {"message": {"content": "<reasoning>secret</reasoning>Visible"}}
            ]
        }

        result = await self.stripper.async_post_call_success_hook(
            data={},
            user_api_key_dict=None,
            response=response,
        )

        assert result["choices"][0]["message"]["content"] == "Visible"

    @pytest.mark.asyncio
    async def test_post_hook_preserves_inline_whitespace_after_reasoning(self):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "alpha<reasoning>x</reasoning> beta"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = await self.stripper.async_post_call_success_hook(
            data={},
            user_api_key_dict=None,
            response=mock_response,
        )

        assert result.choices[0].message.content == "alpha beta"

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

    def test_preserves_visible_whitespace_after_inline_reasoning(self):
        stream_filter = ReasoningStreamFilter("reasoning")

        visible = stream_filter.feed("alpha<reasoning>x</reasoning> beta")
        visible += stream_filter.flush()

        assert visible == "alpha beta"

    def test_suppresses_initial_whitespace_split_across_chunks(self):
        stream_filter = ReasoningStreamFilter("reasoning")

        visible = stream_filter.feed("<reasoning>private notes</reasoning>")
        visible += stream_filter.feed("\n")
        visible += stream_filter.feed("\nThe answer is 42.")
        visible += stream_filter.flush()

        assert visible == "The answer is 42."


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

    @pytest.mark.asyncio
    async def test_streaming_hook_flushes_tail_when_finish_chunk_has_no_delta(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Literal <rea"}}]},
            {"choices": [{"index": 0, "finish_reason": "stop"}]},
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_choice_content(choice) or "" for chunk in result for choice in chunk["choices"]) == "Literal <rea"
        assert result[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_hook_flushes_tail_when_object_finish_chunk_has_no_delta(self):
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="Literal <rea"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(index=0, finish_reason="stop")]
            ),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_chunk_content(chunk) or "" for chunk in result) == "Literal <rea"
        assert result[-1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_streaming_hook_strips_dict_shaped_choice_delta(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "content": "<reasoning>x</reasoning>Visible",
                        },
                    }
                ]
            ),
            SimpleNamespace(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_chunk_content(chunk) or "" for chunk in result) == "Visible"
        assert result[-1].choices[0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_hook_preserves_role_only_chunk_without_content(self):
        chunks = [
            _stream_chunk(role="assistant"),
            _stream_chunk("Visible"),
            _stream_chunk(finish_reason="stop"),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert getattr(result[0].choices[0].delta, "role", None) == "assistant"
        assert not hasattr(result[0].choices[0].delta, "content")
        assert "".join(_chunk_content(chunk) or "" for chunk in result) == "Visible"

    @pytest.mark.asyncio
    async def test_streaming_hook_keeps_separate_state_per_choice_index(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    {"index": 0, "delta": {"content": "<rea"}},
                    {"index": 1, "delta": {"content": "Alpha "}},
                ]
            ),
            SimpleNamespace(
                choices=[
                    {"index": 1, "delta": {"content": "Beta"}},
                    {
                        "index": 0,
                        "delta": {"content": "soning>hidden</reasoning>Visible"},
                    },
                ]
            ),
            SimpleNamespace(
                choices=[
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                    {"index": 1, "delta": {}, "finish_reason": "stop"},
                ]
            ),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)
        contents_by_index = {0: [], 1: []}
        for chunk in result:
            for choice in chunk.choices:
                contents_by_index[choice["index"]].append(_choice_content(choice) or "")

        assert "".join(contents_by_index[0]) == "Visible"
        assert "".join(contents_by_index[1]) == "Alpha Beta"

    @pytest.mark.asyncio
    async def test_streaming_hook_strips_content_when_finish_reason_lookup_raises(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    _ChoiceWithBrokenFinishReason(
                        "<reasoning>x</reasoning>Visible"
                    )
                ]
            ),
        ]

        result = await _collect_streaming_hook(self.stripper, chunks)

        assert "".join(_chunk_content(chunk) or "" for chunk in result) == "Visible"
