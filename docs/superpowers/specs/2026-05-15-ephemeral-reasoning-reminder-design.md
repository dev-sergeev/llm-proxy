# Ephemeral Reasoning Reminder Design

Date: 2026-05-15

## Goal

The proxy must force a non-reasoning model to produce a `<reasoning>...</reasoning>` block on every model turn, including coding-agent loops where the latest request ends with `assistant` tool calls or `tool` results instead of a `user` message.

The current implementation injects the reasoning protocol into the last historical `user` message. That works for simple chat, but weakens after tool execution because the active end of the prompt is no longer the user request. The model may then answer after tool messages without seeing a fresh, adjacent reasoning requirement.

## Chosen Approach

Use an ephemeral trailing `user` reminder whenever the request does not already end with a `user` message.

`RequestModifier` keeps the existing behavior when the last message is `user`: it removes stale `<system-reminder>...</system-reminder>` blocks from that message and prepends a fresh reminder containing `REASONING_INSTRUCTION`.

When the last message role is anything else, including `assistant` or `tool`, `RequestModifier` appends a new synthetic message:

```json
{"role": "user", "content": "<system-reminder>...</system-reminder>"}
```

This keeps the reasoning protocol adjacent to the model's next generation step without moving it into `system`. The response-side `ReasoningStripper` remains responsible only for removing model-produced `<reasoning>...</reasoning>` content from non-streaming and streaming responses.

## Components

### `RequestModifier`

Owns request mutation in `async_pre_call_hook` for `completion`, `acompletion`, and `text_completion`.

It will:

- skip all non-completion call types;
- apply `SYSTEM_PROMPT_DEFAULT` and `USER_PROMPT_APPEND` as today;
- clean stale `<system-reminder>...</system-reminder>` blocks from every historical `user` message to prevent accumulation in long agent loops;
- inject the fresh reasoning reminder into the final message if that final message is `user`;
- append a synthetic trailing `user` reminder if the final message is not `user`;
- append a synthetic trailing `user` reminder when no `user` message exists.

### `ReasoningStripper`

Remains unchanged in responsibility.

It will:

- strip closed `<reasoning>...</reasoning>` blocks from non-streaming responses;
- drop unclosed leading reasoning blocks rather than leak private text;
- strip reasoning from streaming `delta.content` chunks while preserving metadata chunks, tool-call chunks, finish reasons, and usage.

### `JsonlLogger`

Remains enabled for verification. The success log should include the mutated request messages and the raw upstream response payload in `logs/requests.jsonl`, allowing manual inspection that the upstream model produced a reasoning block while the client response was sanitized.

## Data Flow

1. A coding agent sends an OpenAI-compatible chat completion request to the proxy.
2. The request may end with `tool` after the agent returns tool output to the model.
3. `RequestModifier.async_pre_call_hook` removes stale reminder blocks from historical user messages.
4. If the request ends with `tool` or `assistant`, the hook appends a synthetic trailing `user` reminder.
5. LiteLLM forwards the mutated request to the configured upstream model.
6. The model sees the fresh trailing reminder and should begin its response with `<reasoning>...</reasoning>`.
7. `ReasoningStripper` removes the reasoning block before returning the response to the client.
8. `JsonlLogger` records enough information to verify the upstream prompt and response behavior.

## Message Shape

For string content, the synthetic reminder message uses a string:

```json
{
  "role": "user",
  "content": "<system-reminder>You MUST begin every response...</system-reminder>"
}
```

For existing final user messages that use content arrays, the current array-preserving behavior remains: the reminder is inserted as the first text block and non-text blocks are preserved in order.

The synthetic trailing reminder does not include the previous user text. Its purpose is only to restate the protocol at the active end of the conversation.

## Error Handling

The hook should be conservative:

- if `messages` is absent or empty, create a single synthetic `user` reminder message;
- if a message has an unexpected content shape, preserve the message and append the synthetic reminder when needed;
- never mutate non-completion requests;
- never duplicate stale reminder blocks across repeated turns;
- never rely on the upstream model's native reasoning features.

The response stripper remains defensive against malformed response objects and unusual streaming chunks.

## Tests

Unit tests should cover:

- appending a synthetic trailing `user` reminder when the final role is `tool`;
- appending a synthetic trailing `user` reminder when the final role is `assistant`;
- preserving current behavior when the final role is `user`;
- cleaning stale reminders from older user messages before appending a trailing reminder;
- creating a synthetic reminder when `messages` is empty or has no user messages;
- preserving existing response stripping behavior.

End-to-end tests should cover:

- a proxy request shaped like `user -> assistant tool_calls -> tool`;
- the upstream payload ending with the synthetic `user` reminder;
- the upstream response containing `<reasoning>...</reasoning>`;
- the client-visible response containing only the final answer;
- `logs/requests.jsonl` showing evidence that the raw upstream response contained reasoning.

Manual verification should run a real coding-agent style request against Qwen via OpenRouter through the proxy, then inspect the proxy logs to confirm that a reasoning block appeared upstream and was stripped before reaching the client.

## Out of Scope

This design does not move the reasoning protocol into `system`, add per-model policies, expose hidden reasoning to clients, or attempt to validate the semantic quality of the reasoning. It only makes the reminder placement robust for old non-reasoning models in tool-loop conversations and verifies that the model emits a removable reasoning block.
