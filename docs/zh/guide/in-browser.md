---
description: '使用 createWebLLM 在浏览器中通过 WebGPU 运行 Llama、Qwen、SmolLM 等开源模型：无需 API key，无需服务器，默认保护隐私。'
---

# 浏览器内模型

`createWebLLM` 借助 [WebLLM](https://github.com/mlc-ai/web-llm)，直接在浏览器中通过 **WebGPU** 运行开源模型（Llama、Qwen、SmolLM、Phi、Gemma……）。请求不会离开设备，因此你不需要 API key、服务器，也无需按 token 付费。

[在在线体验中试试 →](../playground)

## 安装 {#install}

WebLLM 是一个可选的 peer 依赖：仅在使用该提供方时才需要安装。

```bash
npm i @mlc-ai/web-llm
```

## 用法 {#usage}

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

第一次请求会下载并编译模型。浏览器会缓存模型权重，之后再次访问只需几秒即可启动。WebLLM `prebuiltAppConfig` 中的任意模型 ID 都可以使用，也可以通过 `engineConfig.appConfig` 添加你自己的模型。

## 加载进度 {#loading-progress}

模型体积有数百 MB，因此应展示加载进度，并在第一条消息之前提前加载：

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

| 成员 | 说明 |
|--------|-------------|
| `load(model)` | 下载并编译模型。并发调用会被去重。 |
| `unload()` | 释放 GPU 内存。下一次请求会重新加载模型。 |
| `loadedModel` | 当前位于 GPU 上的模型 |

## 在主线程之外运行 {#run-off-the-main-thread}

推理主要受 GPU 限制，但分词和调度仍然在 JavaScript 中运行。将引擎移到 Web Worker 中，可以保持 UI 流畅：

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

## 切换模型 {#switching-models}

在每个组合式函数中，`model` 都接受 ref 或 getter。提供方会在下一次请求时卸载旧模型并加载新模型：

```ts
const model = ref('SmolLM2-360M-Instruct-q4f16_1-MLC')
const chat = useChat({ provider: ai, model })

model.value = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
```

## 16 位与 32 位模型 {#16-bit-and-32-bit-models}

大多数模型提供两种构建：`q4f16`（更小、更快）和 `q4f32`。16 位构建需要 WebGPU 的 `shader-f16` 特性。如果不支持，或显卡驱动有问题，模型虽然能加载，但会输出乱码。最常见的是 Intel 显卡：它们声明支持 `shader-f16`，但 16 位模型会输出乱码。

`createWebLLM` 会自动处理。传入 `q4f16` 的 id，在无法可靠运行 16 位的显卡上，提供方会加载同一模型的 `q4f32` 构建：

```ts
const ai = createWebLLM({ createEngine: CreateMLCEngine })
await ai.load('Llama-3.2-1B-Instruct-q4f16_1-MLC')
ai.loadedModel // 'Llama-3.2-1B-Instruct-q4f32_1-MLC' 在 f16 不可靠的显卡上
```

`precision` 选项可以覆盖检测：`'f16'` 或 `'f32'` 总是加载对应构建，默认值为 `'auto'`。如果用户仍然看到乱码，可以像[在线体验](../playground)那样提供切换到 32 位的开关：

```ts
createWebLLM({ createEngine: CreateMLCEngine, precision: force32.value ? 'f32' : 'auto' })
```

`detectWebGPUPrecision()` 返回与提供方相同的判断结果，例如可在加载前显示正确的下载大小。

## 选择模型 {#choosing-a-model}

| 模型 | 下载大小 | 备注 |
|-------|----------|-------|
| `SmolLM2-360M-Instruct-q4f16_1-MLC` | ~380 MB | 加载最快，适合简单任务 |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | ~950 MB | 综合表现不错 |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~880 MB | 回答质量更好 |
| `Qwen3-0.6B-q4f16_1-MLC` | ~1.4 GB | 以 `reasoning` part 流式输出推理过程 |
| `Hermes-3-Llama-3.2-3B-q4f16_1-MLC` | ~2.3 GB | 工具调用 |
| `Phi-3.5-vision-instruct-q4f16_1-MLC` | ~4.5 GB | 图片输入 |

## 功能 {#features}

- **流式输出**、`stop()`（会中断 GPU 上的生成）以及 token `usage`。
- **推理：** Qwen3 和 DeepSeek-R1 蒸馏模型输出的 `<think>…</think>` 块会以 `reasoning` part 的形式流式输出。
- **工具：** `tools` 和 `onToolCall` 可用于 Hermes 等支持函数调用的模型。
- **附件：** 视觉模型支持图片，文本文件会被内联。其他文件类型会抛出错误。
- **选项：** `providerOptions` 接受 `temperature`、`top_p`、`max_tokens`（或 `maxTokens`）、`frequency_penalty`、`presence_penalty`、`seed`、`stop`、`logit_bias`、`response_format` 和 `extra_body`。

## 浏览器支持 {#browser-support}

桌面端 Chrome 和 Edge 113+、Android 上的 Chrome 以及 Safari 26+ 已内置 WebGPU。在提供该功能前请先检测支持情况：

```ts
import { createSSEProvider, isWebGPUSupported } from 'vue-llm/providers'

const ai = isWebGPUSupported()
  ? createWebLLM({ createEngine: CreateMLCEngine })
  : createSSEProvider({ url: '/api/chat' }) // fall back to your server
```

缺少 WebGPU 时，`validateConfig()` 会抛出清晰的错误，`useChat` 会通过 `error` 暴露该错误。

::: tip 内存
模型所需的 GPU 内存大致等于其下载大小。在内存较小的设备上，请选择更小的模型，或调低 `chatOptions.context_window_size`。
:::
