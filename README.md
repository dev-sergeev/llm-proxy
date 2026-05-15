# super_proxy

LiteLLM proxy в роли OpenAI-совместимого шлюза с тремя независимыми дополнениями:
1. **`RequestModifier`** — `async_pre_call_hook`, добавляет system-промт, при необходимости дополняет последнее user-сообщение и внедряет reasoning-instruction в последнее `user`-сообщение.
2. **`ReasoningStripper`** — `async_post_call_success_hook` + `async_post_call_streaming_iterator_hook`: вырезает `<reasoning>...</reasoning>` из non-streaming и streaming-ответов перед отдачей клиенту.
3. **`JsonlLogger`** — `async_log_success_event` / `async_log_failure_event`, пишет логи построчно в `logs/requests.jsonl`.

Любое имя модели на входе (`model: "*"` в `config.yaml`) пересылается в один **hosted vLLM**-бэкенд. Каждый компонент включается/выключается отдельной строкой в `config.yaml` (`litellm_settings.callbacks`).

## Структура проекта

```
super_proxy/
├── config.yaml             # маршрутизация на vLLM + регистрация callback'ов
├── custom_callbacks.py     # RequestModifier, ReasoningStripper и JsonlLogger
├── requirements.txt        # litellm[proxy]>=1.83.0
├── .env.example            # шаблон секретов (по умолчанию пуст)
├── docs/
│   └── pre_call_hook_args.md   # справочник по аргументам async_pre_call_hook
└── logs/                   # JSONL-логи, директория создаётся на старте
    └── requests.jsonl
```

Требуется Python ≥ 3.9.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Конфигурация

1. В `config.yaml` подставь реальное имя модели vLLM вместо `MODEL_PLACEHOLDER` (то же значение, что в `vllm serve --model …`) и при необходимости поправь `api_base`.
2. Если vLLM поднят с `--api-key`:

```bash
cp .env.example .env
# раскомментировать VLLM_API_KEY и положить реальный токен
```

И в `config.yaml` добавь под `litellm_params`:
```yaml
api_key: os.environ/VLLM_API_KEY
```

## Запуск

```bash
set -a && source .env && set +a   # можно пропустить, если .env пуст
litellm --config config.yaml --port 4000
```

## Пример запроса

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"any-name-you-like","messages":[{"role":"user","content":"Hello"}]}' | jq .
```

Имя модели в `model` можно слать любое — wildcard `*` отправит запрос на vLLM. После ответа в `logs/requests.jsonl` появится строка с полным [Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec) — `messages` (включая модифицированное последнее `user`-сообщение), `response`, `response_cost`, `total_tokens`, тайминги, `error_information` и т.д.

## Где править правила модификации

В `custom_callbacks.py`, секция «Модификация запросов» — две строковые константы:

- **`SYSTEM_PROMPT_DEFAULT`** — system-промт, который встаёт первым.
  Если у клиента уже есть system-сообщение, наш текст дописывается в **начало** его содержимого.
  Пустая строка — функция выключена.
- **`USER_PROMPT_APPEND`** — текст, дописываемый в **конец** последнего user-сообщения.
  Пустая строка — функция выключена.

Изменения подхватываются при перезапуске proxy.

Подробнее про аргументы, передаваемые в хук — [docs/pre_call_hook_args.md](docs/pre_call_hook_args.md).

## Reasoning-стриппер

В `custom_callbacks.py`, секция «Удаление reasoning-тега из ответа модели»:

- **`REASONING_TAG`** — имя XML-тега (по умолчанию `reasoning`).
- **`REASONING_INSTRUCTION`** — текст reasoning-протокола, который `RequestModifier` добавляет в последнее `user`-сообщение. Для строкового `content` reminder встраивается в строку напрямую. Для `content`-массивов callback добавляет первый text-блок с reminder и сохраняет остальные блоки.

Reasoning-instruction впрыскивается для `call_type` из `completion`, `text_completion`, `acompletion`. Это покрывает живой proxy path LiteLLM, где chat completion обычно проходит как `acompletion`.

`ReasoningStripper` только очищает ответы модели: удаляет `<reasoning>...</reasoning>` из non-streaming ответов и из streaming-чанков. Потоковая очистка работает с тегами, разорванными между чанками, и сохраняет служебные чанки с `role`, `tool_calls`, `usage` и `finish_reason`.

Если модель открыла `<reasoning>`, но не закрыла тег, содержимое reasoning не отдаётся клиенту.

## Как отключить компонент

В `config.yaml` закомментируй строку в `litellm_settings.callbacks`:

```yaml
litellm_settings:
  callbacks:
    - custom_callbacks.request_modifier   # ← закомментируй, чтобы перестать менять запросы
    - custom_callbacks.reasoning_stripper # ← закомментируй, чтобы перестать вырезать <reasoning>…</reasoning>
    - custom_callbacks.jsonl_logger       # ← закомментируй, чтобы перестать писать .jsonl
```
