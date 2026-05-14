import pytest
from unittest.mock import MagicMock
from custom_callbacks import ReasoningStripper, REASONING_INSTRUCTION


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
