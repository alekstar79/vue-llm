---
description: 'Run Llama, Qwen, SmolLM and other open models in the browser on WebGPU with createWebLLM: no API key, no server, private by default.'
---

# In-Browser Models

`createWebLLM` runs open models (Llama, Qwen, SmolLM, Phi, Gemma…) directly in the browser on **WebGPU** through [WebLLM](https://github.com/mlc-ai/web-llm). Requests never leave the device, so you need no API key, server or per-token bill.

[Try it in the playground →](../playground)

## Install

WebLLM is an optional peer: install it only if you use this provider.

```bash
npm i @mlc-ai/web-llm
```

## Usage

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

The first request downloads and compiles the model. The browser caches the weights, so later visits start in seconds. Any model ID from WebLLM's `prebuiltAppConfig` works, and `engineConfig.appConfig` adds your own.

## Loading progress

Models are hundreds of megabytes, so show progress and load ahead of the first message:

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

| Member | Description |
|--------|-------------|
| `load(model)` | Download and compile a model. Concurrent calls are deduplicated. |
| `unload()` | Free GPU memory. The next request loads the model again. |
| `loadedModel` | Model currently on the GPU |

## Run off the main thread

Inference is GPU-bound, but tokenization and scheduling still run in JavaScript. Move the engine to a Web Worker to keep the UI smooth:

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

## Switching models

`model` accepts a ref or getter in every composable. The provider unloads the old model and loads the new one on the next request:

```ts
const model = ref('SmolLM2-360M-Instruct-q4f16_1-MLC')
const chat = useChat({ provider: ai, model })

model.value = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
```

## 16-bit and 32-bit models

Most models ship two builds: `q4f16` (smaller and faster) and `q4f32`. The 16-bit build needs the WebGPU `shader-f16` feature. Without it, or with buggy GPU drivers, the model loads but answers with gibberish. Intel GPUs are the common case: they report `shader-f16`, yet return gibberish from 16-bit models.

`createWebLLM` handles this for you. Pass a `q4f16` id, and on a GPU that can't run it reliably the provider loads the `q4f32` build of the same model:

```ts
const ai = createWebLLM({ createEngine: CreateMLCEngine })
await ai.load('Llama-3.2-1B-Instruct-q4f16_1-MLC')
ai.loadedModel // 'Llama-3.2-1B-Instruct-q4f32_1-MLC' on a GPU without reliable f16
```

The `precision` option overrides the check: `'f16'` or `'f32'` always loads that build, `'auto'` is the default. If users still see gibberish, give them a 32-bit switch, as the [playground](../playground) does:

```ts
createWebLLM({ createEngine: CreateMLCEngine, precision: force32.value ? 'f32' : 'auto' })
```

`detectWebGPUPrecision()` returns the same answer the provider uses, for example to show the right download size before loading.

## Choosing a model

| Model | Download | Notes |
|-------|----------|-------|
| `SmolLM2-360M-Instruct-q4f16_1-MLC` | ~380 MB | Fastest to load, simple tasks |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | ~950 MB | Good all-rounder |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~880 MB | Better answers |
| `Qwen3-0.6B-q4f16_1-MLC` | ~1.4 GB | Streams its reasoning as `reasoning` parts |
| `Hermes-3-Llama-3.2-3B-q4f16_1-MLC` | ~2.3 GB | Tool calling |
| `Phi-3.5-vision-instruct-q4f16_1-MLC` | ~4.5 GB | Image input |

## Features

- **Streaming**, `stop()` (interrupts generation on the GPU) and token `usage`.
- **Reasoning:** `<think>…</think>` blocks from Qwen3 and DeepSeek-R1 distills are streamed as `reasoning` parts.
- **Tools:** `tools` and `onToolCall` work with function-calling models such as Hermes.
- **Attachments:** images for vision models, and text files are inlined. Other file types throw.
- **Options:** `providerOptions` accepts `temperature`, `top_p`, `max_tokens` (or `maxTokens`), `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `logit_bias`, `response_format` and `extra_body`.

## Browser support

WebGPU ships in Chrome and Edge 113+ on desktop, Chrome on Android, and Safari 26+. Check support before offering the feature:

```ts
import { createSSEProvider, isWebGPUSupported } from 'vue-llm/providers'

const ai = isWebGPUSupported()
  ? createWebLLM({ createEngine: CreateMLCEngine })
  : createSSEProvider({ url: '/api/chat' }) // fall back to your server
```

`validateConfig()` throws a clear error when WebGPU is missing, which `useChat` surfaces through `error`.

::: tip Memory
A model needs roughly its download size in GPU memory. On low-memory devices pick a smaller model or lower `chatOptions.context_window_size`.
:::
