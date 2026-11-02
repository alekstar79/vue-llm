---
description: 'Ejecuta Llama, Qwen, SmolLM y otros modelos abiertos en el navegador sobre WebGPU con createWebLLM: sin API key, sin servidor y privado por defecto.'
---

# Modelos en el navegador

`createWebLLM` ejecuta modelos abiertos (Llama, Qwen, SmolLM, Phi, Gemma…) directamente en el navegador sobre **WebGPU** mediante [WebLLM](https://github.com/mlc-ai/web-llm). Las peticiones nunca salen del dispositivo, así que no necesitas API key, servidor ni pagar por token.

[Pruébalo en el playground →](../playground)

## Instalación {#install}

WebLLM es una peer dependency opcional: instálala solo si usas este proveedor.

```bash
npm i @mlc-ai/web-llm
```

## Uso {#usage}

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

La primera petición descarga y compila el modelo. El navegador guarda los pesos en caché, así que en las siguientes visitas arranca en segundos. Funciona cualquier ID de modelo del `prebuiltAppConfig` de WebLLM, y `engineConfig.appConfig` permite añadir los tuyos.

## Progreso de carga {#loading-progress}

Los modelos ocupan cientos de megabytes, así que muestra el progreso y cárgalos antes del primer mensaje:

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

| Miembro | Descripción |
|--------|-------------|
| `load(model)` | Descarga y compila un modelo. Las llamadas concurrentes se deduplican. |
| `unload()` | Libera la memoria de la GPU. La siguiente petición vuelve a cargar el modelo. |
| `loadedModel` | Modelo cargado actualmente en la GPU |

## Fuera del hilo principal {#run-off-the-main-thread}

La inferencia depende de la GPU, pero la tokenización y la planificación siguen ejecutándose en JavaScript. Mueve el motor a un Web Worker para que la UI siga fluida:

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

## Cambiar de modelo {#switching-models}

`model` acepta un ref o un getter en todos los composables. El proveedor descarga el modelo anterior y carga el nuevo en la siguiente petición:

```ts
const model = ref('SmolLM2-360M-Instruct-q4f16_1-MLC')
const chat = useChat({ provider: ai, model })

model.value = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
```

## Modelos de 16 y 32 bits {#16-bit-and-32-bit-models}

La mayoría de los modelos vienen en dos versiones: `q4f16` (más pequeña y rápida) y `q4f32`. La versión de 16 bits necesita la función `shader-f16` de WebGPU. Sin ella, o con drivers de GPU defectuosos, el modelo carga pero responde con texto sin sentido. El caso más común son las GPU Intel: anuncian `shader-f16`, pero los modelos de 16 bits devuelven texto sin sentido.

`createWebLLM` se encarga de esto. Pasa un id `q4f16` y, en una GPU que no lo ejecuta de forma fiable, el proveedor carga la versión `q4f32` del mismo modelo:

```ts
const ai = createWebLLM({ createEngine: CreateMLCEngine })
await ai.load('Llama-3.2-1B-Instruct-q4f16_1-MLC')
ai.loadedModel // 'Llama-3.2-1B-Instruct-q4f32_1-MLC' en una GPU sin f16 fiable
```

La opción `precision` anula la comprobación: `'f16'` o `'f32'` siempre cargan esa versión, `'auto'` es el valor por defecto. Si los usuarios siguen viendo texto sin sentido, ofrece un interruptor a 32 bits, como hace el [playground](../playground):

```ts
createWebLLM({ createEngine: CreateMLCEngine, precision: force32.value ? 'f32' : 'auto' })
```

`detectWebGPUPrecision()` devuelve la misma respuesta que usa el proveedor, por ejemplo para mostrar el tamaño de descarga correcto antes de cargar.

## Elegir un modelo {#choosing-a-model}

| Modelo | Descarga | Notas |
|-------|----------|-------|
| `SmolLM2-360M-Instruct-q4f16_1-MLC` | ~380 MB | El que carga más rápido, tareas sencillas |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | ~950 MB | Buen todoterreno |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~880 MB | Mejores respuestas |
| `Qwen3-0.6B-q4f16_1-MLC` | ~1.4 GB | Transmite su razonamiento como partes `reasoning` |
| `Hermes-3-Llama-3.2-3B-q4f16_1-MLC` | ~2.3 GB | Llamadas a herramientas |
| `Phi-3.5-vision-instruct-q4f16_1-MLC` | ~4.5 GB | Entrada de imágenes |

## Funciones {#features}

- **Streaming**, `stop()` (interrumpe la generación en la GPU) y `usage` de tokens.
- **Razonamiento:** los bloques `<think>…</think>` de Qwen3 y de los destilados de DeepSeek-R1 se transmiten como partes `reasoning`.
- **Herramientas:** `tools` y `onToolCall` funcionan con modelos con function calling como Hermes.
- **Adjuntos:** imágenes para modelos de visión; los archivos de texto se insertan en línea. Otros tipos de archivo lanzan un error.
- **Opciones:** `providerOptions` acepta `temperature`, `top_p`, `max_tokens` (o `maxTokens`), `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `logit_bias`, `response_format` y `extra_body`.

## Compatibilidad con navegadores {#browser-support}

WebGPU está disponible en Chrome y Edge 113+ en escritorio, Chrome en Android y Safari 26+. Comprueba la compatibilidad antes de ofrecer la función:

```ts
import { createSSEProvider, isWebGPUSupported } from 'vue-llm/providers'

const ai = isWebGPUSupported()
  ? createWebLLM({ createEngine: CreateMLCEngine })
  : createSSEProvider({ url: '/api/chat' }) // fall back to your server
```

`validateConfig()` lanza un error claro cuando falta WebGPU, y `useChat` lo expone a través de `error`.

::: tip Memoria
Un modelo necesita aproximadamente su tamaño de descarga en memoria de GPU. En dispositivos con poca memoria, elige un modelo más pequeño o reduce `chatOptions.context_window_size`.
:::
