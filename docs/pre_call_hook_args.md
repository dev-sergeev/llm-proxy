# Аргументы `async_pre_call_hook`

Хук вызывается LiteLLM proxy **сразу перед отправкой запроса в LLM-провайдера** (после аутентификации, валидации и model-routing). Метод асинхронный, должен вернуть `data: dict` с изменениями, либо строку-причину отказа (тогда proxy ответит клиенту ошибкой).

```python
async def async_pre_call_hook(
    self,
    user_api_key_dict: UserAPIKeyAuth,
    cache: DualCache,
    data: dict,
    call_type: Literal[
        "completion", "text_completion", "embeddings",
        "image_generation", "moderation", "audio_transcription",
    ],
) -> Optional[dict | str]: ...
```

Ниже — примерные значения, какие можно ожидать на входе.

---

## 1. `user_api_key_dict: UserAPIKeyAuth`

Pydantic-модель из `litellm.proxy._types`. Описывает виртуальный API-ключ, с которым пришёл запрос. Если `master_key` не задан и proxy открыт — большинство полей `None`/пустые, но объект всё равно передаётся.

Пример (после `.model_dump()`):

```python
{
    "api_key":   "sk-virtual-abc123",         # hash / короткое представление
    "user_id":   "alice@example.com",
    "team_id":   "team-engineering",
    "key_alias": "team-support",              # ← по этому полю ищем в SYSTEM_BY_KEY_ALIAS
    "key_name":  "support bot key",
    "models":    ["*"],                        # список разрешённых моделей; "*" — все
    "spend":     0.0123,                       # USD, потрачено этим ключом
    "max_budget": 100.0,
    "tpm_limit": None,
    "rpm_limit": None,
    "metadata":  {"environment": "prod"},
    "permissions": {},
    "litellm_budget_table": None,
}
```

В коде доступ — через атрибут, не индексирование:

```python
alias = getattr(user_api_key_dict, "key_alias", None)
user  = getattr(user_api_key_dict, "user_id",   None)
```

---

## 2. `cache: DualCache`

**Не данные**, а живой объект кеша (in-memory + опциональный Redis). Один на весь процесс proxy. Используется для дедупликации, rate-limit'ов, переноса состояния между несколькими хуками одного запроса.

Полезные методы:

```python
await cache.async_get_cache(key="my-key")
await cache.async_set_cache(key="my-key", value={"x": 1}, ttl=60)
await cache.async_increment_cache(key="counter", value=1, ttl=60)
```

В простой модификации запроса обычно не нужен.

---

## 3. `data: dict`

Тело запроса от клиента **плюс** метаданные, проставленные proxy. То, что мы тут возвращаем — уйдёт в LLM-провайдера.

Пример для `call_type="completion"`:

```python
{
    # ===== поля, которые прислал клиент =====
    "model":       "any-name-you-like",      # имя ДО wildcard-маппинга
    "messages": [
        {"role": "user", "content": "Привет!"},
    ],
    "temperature": 0.7,
    "max_tokens":  1000,
    "stream":      False,
    "stream_options": None,
    "tools":        None,
    "tool_choice":  None,
    "response_format": None,
    "n":           1,
    "stop":        None,

    # ===== поля, которые добавил proxy =====
    "metadata": {
        "headers": {                          # HTTP-заголовки клиента
            "host": "localhost:4000",
            "user-agent": "curl/8.5.0",
            "content-type": "application/json",
        },
        "user_api_key":            "sk-virtual-abc123",
        "user_api_key_alias":      "team-support",
        "user_api_key_user_id":    "alice@example.com",
        "user_api_key_team_id":    "team-engineering",
        "user_api_key_metadata":   {},
        "endpoint":   "http://localhost:4000/v1/chat/completions",
        "model_group": "*",                   # имя из model_list (у нас wildcard)
        "deployment": "hosted_vllm/MODEL_PLACEHOLDER",
    },
    "proxy_server_request": {
        "url":     "http://localhost:4000/v1/chat/completions",
        "method":  "POST",
        "headers": {...},
        "body":    {                          # точное тело как пришло
            "model": "any-name-you-like",
            "messages": [...],
        },
    },
    "litellm_call_id":     "8c1e...-uuid",
    "litellm_logging_obj": "<LiteLLMLogging object>",
}
```

**Что можно безопасно менять прямо в `data`:**

| Поле | Эффект |
|------|--------|
| `data["model"]` | Перенаправить на другую модель (раньше, чем wildcard) |
| `data["messages"]` | Добавить/изменить сообщения (то, что делаем мы) |
| `data["temperature"]`, `data["max_tokens"]`, … | Параметры генерации |
| `data["tools"]`, `data["tool_choice"]` | Tool calling |
| `data["response_format"]` | JSON-mode / структурированный ответ |

**Возврат:**
- `return data` — отправить изменённый запрос дальше
- `return "reason text"` — отказ (proxy ответит клиенту ошибкой с этим текстом)

---

## 4. `call_type`

Строковый литерал, тип эндпоинта:

| Значение | Эндпоинт | Что в `data` |
|---|---|---|
| `"completion"`         | `/v1/chat/completions`     | `messages` |
| `"text_completion"`    | `/v1/completions`          | `prompt` (строка) |
| `"embeddings"`         | `/v1/embeddings`           | `input` |
| `"image_generation"`   | `/v1/images/generations`   | `prompt` |
| `"moderation"`         | `/v1/moderations`          | `input` |
| `"audio_transcription"`| `/v1/audio/transcriptions` | `file` |

Полезно, чтобы пропустить модификацию для embeddings/moderation, где `messages` нет — мы так и делаем:

```python
if call_type not in ("completion", "text_completion"):
    return data
```

---

## Как получить РЕАЛЬНЫЙ дамп со своего proxy

Если хочется увидеть точные значения именно из своей установки — добавь в `async_pre_call_hook` одноразовый дамп:

```python
import json
from pathlib import Path

DUMP = Path("logs/pre_call_dump.json")
if not DUMP.exists():
    DUMP.write_text(json.dumps({
        "user_api_key_dict": getattr(user_api_key_dict, "model_dump", lambda: str(user_api_key_dict))(),
        "data": data,
        "call_type": call_type,
    }, ensure_ascii=False, default=str, indent=2))
```

После первого запроса появится `logs/pre_call_dump.json` со всем содержимым. Затем эти строки можно удалить.

---

## Источники

- [LiteLLM call_hooks](https://docs.litellm.ai/docs/proxy/call_hooks) — официальная документация хука
- [LiteLLM logging_spec](https://docs.litellm.ai/docs/proxy/logging_spec) — структура `standard_logging_object` (то, что попадает в наш JSONL после ответа модели)
