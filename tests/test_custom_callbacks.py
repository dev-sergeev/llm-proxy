import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_callbacks import (
    ReasoningStreamFilter,
    ReasoningStripper,
    REASONING_INSTRUCTION,
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


class TestReasoningStripperPreHook:
    """Test ReasoningStripper.async_pre_call_hook message transformation."""

    def setup_method(self):
        self.stripper = ReasoningStripper()

    @pytest.mark.asyncio
    async def test_prepend_instruction_to_string_user_message(self):
        """When last message is a user string, prepend instruction in content array."""
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"}
            ]
        }

        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="completion"
        )

        # Last message should now be content array
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert len(last_msg["content"]) == 2

        # First block should have wrapped instruction
        assert last_msg["content"][0]["type"] == "text"
        assert "<system-reminder>" in last_msg["content"][0]["text"]
        assert "You MUST begin every response" in last_msg["content"][0]["text"]
        assert "</system-reminder>" in last_msg["content"][0]["text"]

        # Second block should have original message
        assert last_msg["content"][1]["type"] == "text"
        assert last_msg["content"][1]["text"] == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_prepend_instruction_to_content_array_message(self):
        """When last message already has content array, prepend instruction block."""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                    ]
                }
            ]
        }

        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="completion"
        )

        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert len(last_msg["content"]) == 3

        # First block: wrapped instruction
        assert last_msg["content"][0]["type"] == "text"
        assert "<system-reminder>" in last_msg["content"][0]["text"]

        # Original blocks preserved
        assert last_msg["content"][1]["type"] == "text"
        assert last_msg["content"][1]["text"] == "Analyze this:"
        assert last_msg["content"][2]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_create_user_message_when_none_exist(self):
        """When no user message exists, create one with wrapped instruction."""
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."}
            ]
        }

        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="completion"
        )

        # Should have added a user message
        assert len(result["messages"]) == 2
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert len(last_msg["content"]) == 1

        # Should contain wrapped instruction
        assert last_msg["content"][0]["type"] == "text"
        assert "<system-reminder>" in last_msg["content"][0]["text"]
        assert "You MUST begin every response" in last_msg["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_skip_non_completion_call_types(self):
        """Should not modify messages for non-completion call types."""
        data = {
            "messages": [
                {"role": "user", "content": "test"}
            ]
        }

        # Test with embeddings call type
        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="embeddings"
        )

        # Message should be unchanged
        assert result["messages"][0]["content"] == "test"

    @pytest.mark.asyncio
    async def test_skip_when_instruction_disabled(self, monkeypatch):
        """Should not modify messages when REASONING_INSTRUCTION is empty."""
        monkeypatch.setattr("custom_callbacks.REASONING_INSTRUCTION", "")

        data = {
            "messages": [
                {"role": "user", "content": "test"}
            ]
        }

        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="completion"
        )

        # Message should be unchanged
        assert result["messages"][0]["content"] == "test"

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
    async def test_cleanup_existing_system_reminder_in_string(self):
        """Should remove existing system-reminder blocks before adding new one."""
        old_reminder = "<system-reminder>Old instruction here</system-reminder>\n\nUser query"
        data = {
            "messages": [
                {"role": "user", "content": old_reminder}
            ]
        }

        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="completion"
        )

        last_msg = result["messages"][-1]
        content = last_msg["content"]

        # Should have content array
        assert isinstance(content, list)
        # Should have exactly 2 blocks: new instruction + original query
        assert len(content) == 2

        # First block: new instruction
        assert "<system-reminder>" in content[0]["text"]
        # Second block: cleaned query (old reminder removed)
        assert content[1]["text"] == "User query"
        assert "<system-reminder>" not in content[1]["text"]

    @pytest.mark.asyncio
    async def test_cleanup_existing_system_reminder_in_array(self):
        """Should remove existing system-reminder blocks from content arrays."""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<system-reminder>Old instruction</system-reminder>\n\nFirst part"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                        {"type": "text", "text": "Final query"}
                    ]
                }
            ]
        }

        result = await self.stripper.async_pre_call_hook(
            user_api_key_dict=None,
            cache=None,
            data=data,
            call_type="completion"
        )

        last_msg = result["messages"][-1]
        content = last_msg["content"]

        # Should have content array
        assert isinstance(content, list)
        # Should have: new instruction + original text blocks (old reminder removed) + image
        assert len(content) == 4

        # First block: new instruction
        assert "<system-reminder>" in content[0]["text"]
        # Second block: first text with old reminder removed
        assert content[1]["text"] == "First part"
        # Third block: image preserved
        assert content[2]["type"] == "image_url"
        # Fourth block: final query
        assert content[3]["text"] == "Final query"


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
