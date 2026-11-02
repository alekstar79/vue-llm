---
description: 'Rode Llama, Qwen, SmolLM e outros modelos abertos no navegador em WebGPU com createWebLLM: sem API key, sem servidor e privado por padrão.'
---

# Modelos no navegador

`createWebLLM` roda modelos abertos (Llama, Qwen, SmolLM, Phi, Gemma…) direto no navegador em **WebGPU** por meio do [WebLLM](https://github.com/mlc-ai/web-llm). As requisições nunca saem do dispositivo, então você não precisa de API key, servidor nem pagar por token.

[Experimente no playground →](../playground)

## Instalação {#install}

O WebLLM é uma peer dependency opcional: instale só se for usar este provedor.

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

A primeira requisição baixa e compila o modelo. O navegador guarda os pesos em cache, então as próximas visitas começam em segundos. Qualquer ID de modelo do `prebuiltAppConfig` do WebLLM funciona, e `engineConfig.appConfig` adiciona os seus.

## Progresso do carregamento {#loading-progress}

Os modelos têm centenas de megabytes, então mostre o progresso e carregue antes da primeira mensagem:

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

| Membro | Descrição |
|--------|-------------|
| `load(model)` | Baixa e compila um modelo. Chamadas concorrentes são deduplicadas. |
| `unload()` | Libera a memória da GPU. A próxima requisição carrega o modelo de novo. |
| `loadedModel` | Modelo que está na GPU no momento |

## Fora da thread principal {#run-off-the-main-thread}

A inferência depende da GPU, mas a tokenização e o agendamento ainda rodam em JavaScript. Mova o engine para um Web Worker para manter a UI fluida:

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

## Trocando de modelo {#switching-models}

`model` aceita um ref ou getter em todos os composables. O provedor descarrega o modelo antigo e carrega o novo na próxima requisição:

```ts
const model = ref('SmolLM2-360M-Instruct-q4f16_1-MLC')
const chat = useChat({ provider: ai, model })

model.value = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
```

## Modelos de 16 e 32 bits {#16-bit-and-32-bit-models}

A maioria dos modelos vem em duas versões: `q4f16` (menor e mais rápida) e `q4f32`. A versão de 16 bits precisa do recurso `shader-f16` do WebGPU. Sem ele, ou com drivers de GPU com bugs, o modelo carrega, mas responde com texto sem sentido. O caso mais comum são as GPUs Intel: elas anunciam `shader-f16`, mas os modelos de 16 bits respondem com texto sem sentido.

`createWebLLM` cuida disso. Passe um id `q4f16` e, numa GPU que não o executa de forma confiável, o provedor carrega a versão `q4f32` do mesmo modelo:

```ts
const ai = createWebLLM({ createEngine: CreateMLCEngine })
await ai.load('Llama-3.2-1B-Instruct-q4f16_1-MLC')
ai.loadedModel // 'Llama-3.2-1B-Instruct-q4f32_1-MLC' numa GPU sem f16 confiável
```

A opção `precision` substitui a verificação: `'f16'` ou `'f32'` sempre carregam essa versão, `'auto'` é o padrão. Se os usuários ainda virem texto sem sentido, ofereça uma opção de 32 bits, como faz o [playground](../playground):

```ts
createWebLLM({ createEngine: CreateMLCEngine, precision: force32.value ? 'f32' : 'auto' })
```

`detectWebGPUPrecision()` retorna a mesma resposta que o provedor usa, por exemplo para mostrar o tamanho de download correto antes de carregar.

## Escolhendo um modelo {#choosing-a-model}

| Modelo | Download | Observações |
|-------|----------|-------|
| `SmolLM2-360M-Instruct-q4f16_1-MLC` | ~380 MB | Carrega mais rápido, tarefas simples |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | ~950 MB | Bom para tudo |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~880 MB | Respostas melhores |
| `Qwen3-0.6B-q4f16_1-MLC` | ~1.4 GB | Faz streaming do raciocínio como partes `reasoning` |
| `Hermes-3-Llama-3.2-3B-q4f16_1-MLC` | ~2.3 GB | Chamada de ferramentas |
| `Phi-3.5-vision-instruct-q4f16_1-MLC` | ~4.5 GB | Entrada de imagens |

## Recursos {#features}

- **Streaming**, `stop()` (interrompe a geração na GPU) e `usage` de tokens.
- **Raciocínio:** blocos `<think>…</think>` do Qwen3 e dos distills do DeepSeek-R1 chegam via streaming como partes `reasoning`.
- **Ferramentas:** `tools` e `onToolCall` funcionam com modelos com function calling, como o Hermes.
- **Anexos:** imagens para modelos de visão, e arquivos de texto são incluídos inline. Outros tipos de arquivo lançam erro.
- **Opções:** `providerOptions` aceita `temperature`, `top_p`, `max_tokens` (ou `maxTokens`), `frequency_penalty`, `presence_penalty`, `seed`, `stop`, `logit_bias`, `response_format` e `extra_body`.

## Suporte dos navegadores {#browser-support}

O WebGPU está disponível no Chrome e no Edge 113+ no desktop, no Chrome para Android e no Safari 26+. Verifique o suporte antes de oferecer o recurso:

```ts
import { createSSEProvider, isWebGPUSupported } from 'vue-llm/providers'

const ai = isWebGPUSupported()
  ? createWebLLM({ createEngine: CreateMLCEngine })
  : createSSEProvider({ url: '/api/chat' }) // fall back to your server
```

`validateConfig()` lança um erro claro quando não há WebGPU, e o `useChat` o expõe via `error`.

::: tip Memória
Um modelo precisa de mais ou menos o tamanho do download em memória de GPU. Em dispositivos com pouca memória, escolha um modelo menor ou reduza `chatOptions.context_window_size`.
:::
