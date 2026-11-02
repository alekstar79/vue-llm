---
description: 'Запускайте Llama, Qwen, SmolLM и другие открытые модели в браузере на WebGPU через createWebLLM: без API-ключа, без сервера, приватно по умолчанию.'
---

# Модели в браузере

`createWebLLM` запускает открытые модели (Llama, Qwen, SmolLM, Phi, Gemma…) прямо в браузере на **WebGPU** через [WebLLM](https://github.com/mlc-ai/web-llm). Запросы не покидают устройство, поэтому не нужны ни API-ключ, ни сервер, ни оплата за токены.

[Попробовать в песочнице →](../playground)

## Установка {#install}

WebLLM — необязательная peer-зависимость: устанавливайте её, только если используете этот провайдер.

```bash
npm i @mlc-ai/web-llm
```

## Использование {#usage}

```ts
import { CreateMLCEngine } from '@mlc-ai/web-llm'
import { useChat } from 'vue-llm'
import { createWebLLM } from 'vue-llm/providers'

const ai = createWebLLM({ createEngine: CreateMLCEngine })

const { messages, sendMessage } = useChat({
  provider: ai,
  model: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
})
```

Первый запрос скачивает и компилирует модель. Браузер кэширует веса, поэтому при следующих визитах всё запускается за секунды. Подойдёт любой ID модели из `prebuiltAppConfig` WebLLM, а через `engineConfig.appConfig` можно добавить свои.

## Прогресс загрузки {#loading-progress}

Модели весят сотни мегабайт, поэтому показывайте прогресс и загружайте модель заранее, до первого сообщения:

```vue
<script setup lang="ts">
const progress = ref(0)
const status = ref('')

const ai = createWebLLM({
  createEngine: CreateMLCEngine,
  onProgress: report => {
    progress.value = report.progress // 0..1
    status.value = report.text
  },
})

const model = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC'
const { messages, input, handleSubmit } = useChat({ provider: ai, model })

onMounted(() => ai.load(model))
</script>
```

| Член | Описание |
|--------|-------------|
| `load(model)` | Скачивает и компилирует модель. Параллельные вызовы объединяются. |
| `unload()` | Освобождает память GPU. Следующий запрос загрузит модель заново. |
| `loadedModel` | Модель, которая сейчас загружена в GPU |

## Вне основного потока {#run-off-the-main-thread}

Инференс упирается в GPU, но токенизация и планирование всё равно выполняются в JavaScript. Перенесите движок в Web Worker, чтобы интерфейс не подтормаживал:

```ts
// webllm.worker.ts
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (message: MessageEvent) => handler.onmessage(message)
```

```ts
import { CreateWebWorkerMLCEngine } from '@mlc-ai/web-llm'

const ai = createWebLLM({
  createEngine: (model, engineConfig, chatOptions) => CreateWebWorkerMLCEngine(
    new Worker(new URL('./webllm.worker.ts', import.meta.url), { type: 'module' }),
    model, engineConfig, chatOptions,
  ),
})
```

## Переключение моделей {#switching-models}

Во всех композаблах `model` принимает ref или геттер. Провайдер выгрузит старую модель и загрузит новую при следующем запросе:

```ts
const model = ref('SmolLM2-360M-Instruct-q4f16_1-MLC')
const chat = useChat({ provider: ai, model })

model.value = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
```

## 16-битные и 32-битные модели {#16-bit-and-32-bit-models}

Большинство моделей есть в двух сборках: `q4f16` (меньше и быстрее) и `q4f32`. Для 16-битной нужна возможность WebGPU `shader-f16`. Без неё или с кривыми драйверами видеокарты модель загружается, но отвечает бессмыслицей. Чаще всего это видеокарты Intel: они сообщают о `shader-f16`, но 16-битные модели выдают на них мусор.

`createWebLLM` решает это сам. Передайте id с `q4f16`, и на видеокарте, которая не тянет 16 бит надёжно, провайдер загрузит `q4f32`-сборку той же модели:

```ts
const ai = createWebLLM({ createEngine: CreateMLCEngine })
await ai.load('Llama-3.2-1B-Instruct-q4f16_1-MLC')
ai.loadedModel // 'Llama-3.2-1B-Instruct-q4f32_1-MLC' на видеокарте без надёжного f16
```

Опция `precision` отменяет проверку: `'f16'` или `'f32'` всегда загружают эту сборку, по умолчанию `'auto'`. Если у пользователей всё равно бессмыслица, дайте переключатель на 32 бита, как в [песочнице](../playground):

```ts
createWebLLM({ createEngine: CreateMLCEngine, precision: force32.value ? 'f32' : 'auto' })
```

`detectWebGPUPrecision()` возвращает тот же ответ, что использует провайдер, например чтобы показать правильный размер загрузки заранее.

## Выбор модели {#choosing-a-model}

| Модель | Размер | Примечания |
|-------|----------|-------|
| `SmolLM2-360M-Instruct-q4f16_1-MLC` | ~380 МБ | Загружается быстрее всех, для простых задач |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | ~950 МБ | Хороший универсальный вариант |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~880 МБ | Ответы качественнее |
| `Qwen3-0.6B-q4f16_1-MLC` | ~1.4 ГБ | Стримит рассуждения как части `reasoning` |
| `Hermes-3-Llama-3.2-3B-q4f16_1-MLC` | ~2.3 ГБ | Вызов инструментов |
| `Phi-3.5-vision-instruct-q4f16_1-MLC` | ~4.5 ГБ | Работа с изображениями |

## Возможности {#features}

- **Стриминг**, `stop()` (прерывает генерацию на GPU) и `usage` по токенам.
- **Рассуждения:** блоки `<think>…</think>` от Qwen3 и дистиллятов DeepSeek-R1 стримятся как части `reasoning`.
- **Инструменты:** `tools` и `onToolCall` работают с моделями, поддерживающими function calling, например Hermes.
- **Вложения:** изображения для vision-моделей; текстовые файлы встраиваются в текст. Другие типы файлов вызывают ошибку.
- **Опции:** `providerOptions` принимает `temperature`, `top_p`, `max_tokens` (или `maxTokens`), `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `logit_bias`, `response_format` и `extra_body`.

## Поддержка браузерами {#browser-support}

WebGPU доступен в Chrome и Edge 113+ на десктопе, Chrome на Android и Safari 26+. Проверяйте поддержку, прежде чем предлагать эту возможность:

```ts
import { createSSEProvider, isWebGPUSupported } from 'vue-llm/providers'

const ai = isWebGPUSupported()
  ? createWebLLM({ createEngine: CreateMLCEngine })
  : createSSEProvider({ url: '/api/chat' }) // запасной вариант — ваш сервер
```

Если WebGPU нет, `validateConfig()` выбрасывает понятную ошибку, а `useChat` отдаёт её через `error`.

::: tip Память
Модели нужно примерно столько памяти GPU, сколько весит её загрузка. На устройствах с малым объёмом памяти выбирайте модель поменьше или уменьшайте `chatOptions.context_window_size`.
:::
