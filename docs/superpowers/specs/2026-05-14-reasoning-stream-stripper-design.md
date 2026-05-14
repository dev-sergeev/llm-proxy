# Reasoning Stream Stripper Design

Date: 2026-05-14

## Goal

The proxy should make a non-reasoning hosted model behave more like a reasoning model for agentic clients. For every chat completion request, the proxy injects a reasoning instruction into the last user message. For every response, the proxy removes the model's internal `<reasoning>...</reasoning>` block before the response reaches the client.

The behavior must work for both non-streaming and streaming chat completions. Streaming is required because coding agents such as Gemini CLI or Qwen CLI usually consume token streams.

## Chosen Approach

Use the existing `ReasoningStripper` callback and add streaming cleanup at the iterator level.

Request modification stays in `async_pre_call_hook`: the proxy finds the last `user` message, removes any old `<system-reminder>...</system-reminder>` block from that message, and prepends a fresh `<system-reminder>` text block containing `REASONING_INSTRUCTION`. The proxy does not add the instruction to every historical user message.

Response cleanup uses two paths:

- Non-streaming responses are cleaned in `async_post_call_success_hook`.
- Streaming responses are cleaned by implementing `async_post_call_streaming_iterator_hook`, wrapping the upstream iterator, and filtering `delta.content` chunk by chunk.

## Components

### `ReasoningStripper`

Owns the LiteLLM hooks:

- `async_pre_call_hook` injects the reasoning instruction into the last user message.
- `async_post_call_success_hook` strips reasoning from completed responses.
- `async_post_call_streaming_iterator_hook` strips reasoning from streaming chunks before the proxy serializes them to SSE.

### `ReasoningStreamFilter`

A small stateful helper used only by the streaming hook. It receives text fragments and returns the visible text that can be safely emitted.

It tracks three states:

- outside a reasoning block;
- inside a reasoning block;
- holding a short suffix that might be the beginning of `<reasoning>` or `</reasoning>` split across chunk boundaries.

The helper must never emit text inside a reasoning block. If the model starts `<reasoning>` and never closes it, the buffered/internal content is dropped rather than leaked.

## Data Flow

1. Client sends an OpenAI-compatible `/v1/chat/completions` request to the LiteLLM proxy.
2. `ReasoningStripper.async_pre_call_hook` modifies `data["messages"]`.
3. LiteLLM forwards the request to the configured hosted vLLM backend.
4. For non-streaming responses, `async_post_call_success_hook` edits `choices[*].message.content`.
5. For streaming responses, `async_post_call_streaming_iterator_hook` yields cleaned chunks as they arrive.
6. The client receives only the final answer text, not the internal reasoning block.

## Streaming Behavior

The streaming filter must handle:

- `<reasoning>...</reasoning>` contained in one chunk;
- opening and closing tags split across multiple chunks;
- reasoning text split across many chunks;
- normal answers with no reasoning tag;
- role-only chunks, tool-call chunks, usage chunks, and final finish chunks.

Only textual `choices[*].delta.content` is changed. Non-text streaming metadata is preserved.

If a text chunk becomes empty after stripping, the hook may still yield the chunk when it carries important metadata such as `role`, `tool_calls`, `finish_reason`, or `usage`. Pure empty content-only chunks can be skipped if preserving them adds no value.

## Error Handling

The filtering logic should be conservative:

- Unexpected chunk shapes pass through unchanged.
- Non-string content passes through unchanged.
- Exceptions inside the filter should not crash the stream; the hook should prefer passing through the current chunk unchanged and let LiteLLM handle upstream stream errors normally.

The parser is tag-specific and case-insensitive for the configured `REASONING_TAG`.

## Tests

Unit tests should cover:

- injecting the instruction into string user content;
- injecting the instruction into content-array user content while preserving non-text blocks;
- removing old `<system-reminder>` blocks before adding a new one;
- creating a user message when none exists;
- skipping non-completion call types;
- stripping non-streaming `<reasoning>...</reasoning>`;
- streaming stripping when tags are in one chunk;
- streaming stripping when tags are split across chunks;
- streaming passthrough when no reasoning tag exists;
- preserving final chunks with `finish_reason` or usage metadata.

Manual verification should include a streaming request to the local proxy that returns visible answer text without `<reasoning>` content.

## Out of Scope

This design does not add model-specific prompt routing, per-client configuration, persistence of hidden reasoning, or support for non-chat endpoints. It does not try to guarantee that the hosted model will actually reason well; it only injects the instruction and hides the resulting block when the model follows the requested protocol.
