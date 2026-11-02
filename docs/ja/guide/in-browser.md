---
description: 'createWebLLM を使って、Llama、Qwen、SmolLM などのオープンモデルを WebGPU でブラウザー内で実行します。API キーもサーバーも不要で、デフォルトでプライベートです。'
---

# ブラウザ内モデル

`createWebLLM` は [WebLLM](https://github.com/mlc-ai/web-llm) を通じて、オープンモデル（Llama、Qwen、SmolLM、Phi、Gemma など）をブラウザー内の **WebGPU** 上で直接実行します。リクエストがデバイスの外に出ることはないため、API キーもサーバーもトークン単位の課金も必要ありません。

[プレイグラウンドで試す →](../playground)

## インストール {#install}

WebLLM はオプションの peer dependency です。このプロバイダーを使う場合にのみインストールしてください。

```bash
npm i @mlc-ai/web-llm
```

## 使い方 {#usage}

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

最初のリクエストでモデルがダウンロードされ、コンパイルされます。重みはブラウザーにキャッシュされるため、2 回目以降の訪問では数秒で起動します。WebLLM の `prebuiltAppConfig` にある任意のモデル ID が使え、`engineConfig.appConfig` で独自のモデルを追加することもできます。

## 読み込みの進捗 {#loading-progress}

モデルのサイズは数百 MB あるため、進捗を表示し、最初のメッセージより前に読み込んでおきましょう。

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

| メンバー | 説明 |
|--------|-------------|
| `load(model)` | モデルをダウンロードしてコンパイルします。同時に呼び出された場合は 1 回にまとめられます。 |
| `unload()` | GPU メモリーを解放します。次のリクエストで再びモデルが読み込まれます。 |
| `loadedModel` | 現在 GPU 上にあるモデル |

## メインスレッドの外で実行する {#run-off-the-main-thread}

推論は GPU で行われますが、トークナイズやスケジューリングは JavaScript で実行されます。UI をなめらかに保つには、エンジンを Web Worker に移しましょう。

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

## モデルの切り替え {#switching-models}

どのコンポーザブルでも、`model` には ref やゲッターを渡せます。プロバイダーは次のリクエスト時に古いモデルをアンロードし、新しいモデルを読み込みます。

```ts
const model = ref('SmolLM2-360M-Instruct-q4f16_1-MLC')
const chat = useChat({ provider: ai, model })

model.value = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'
```

## 16 ビットと 32 ビットのモデル {#16-bit-and-32-bit-models}

多くのモデルには `q4f16`（小さく高速）と `q4f32` の 2 種類のビルドがあります。16 ビット版には WebGPU の `shader-f16` 機能が必要です。これがない場合や GPU ドライバーに不具合がある場合、モデルは読み込めても意味不明な回答を返します。よくあるのは Intel GPU で、`shader-f16` を報告するのに 16 ビットモデルが意味不明な回答を返します。

`createWebLLM` がこれを自動で処理します。`q4f16` の id を渡すと、16 ビットを安定して実行できない GPU では、同じモデルの `q4f32` ビルドを読み込みます：

```ts
const ai = createWebLLM({ createEngine: CreateMLCEngine })
await ai.load('Llama-3.2-1B-Instruct-q4f16_1-MLC')
ai.loadedModel // 'Llama-3.2-1B-Instruct-q4f32_1-MLC' f16 が不安定な GPU の場合
```

`precision` オプションでチェックを上書きできます。`'f16'` または `'f32'` は常にそのビルドを読み込み、既定値は `'auto'` です。それでも意味不明な回答が出る場合は、[プレイグラウンド](../playground)のように 32 ビットへの切り替えを用意してください：

```ts
createWebLLM({ createEngine: CreateMLCEngine, precision: force32.value ? 'f32' : 'auto' })
```

`detectWebGPUPrecision()` はプロバイダーと同じ判定結果を返します。読み込み前に正しいダウンロードサイズを表示する場合などに使えます。

## モデルの選び方 {#choosing-a-model}

| モデル | ダウンロード | 備考 |
|-------|----------|-------|
| `SmolLM2-360M-Instruct-q4f16_1-MLC` | ~380 MB | 読み込みが最速、簡単なタスク向け |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | ~950 MB | バランスが良い |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~880 MB | より質の高い回答 |
| `Qwen3-0.6B-q4f16_1-MLC` | ~1.4 GB | 推論過程を `reasoning` パーツとしてストリーミング |
| `Hermes-3-Llama-3.2-3B-q4f16_1-MLC` | ~2.3 GB | ツール呼び出し |
| `Phi-3.5-vision-instruct-q4f16_1-MLC` | ~4.5 GB | 画像入力 |

## 機能 {#features}

- **ストリーミング**、`stop()`（GPU 上の生成を中断）、トークンの `usage`。
- **推論:** Qwen3 や DeepSeek-R1 の蒸留モデルが出力する `<think>…</think>` ブロックは `reasoning` パーツとしてストリーミングされます。
- **ツール:** `tools` と `onToolCall` は、Hermes のような関数呼び出しに対応したモデルで動作します。
- **添付ファイル:** ビジョンモデルでは画像を扱え、テキストファイルはインライン展開されます。それ以外のファイル形式はエラーになります。
- **オプション:** `providerOptions` には `temperature`、`top_p`、`max_tokens`（または `maxTokens`）、`frequency_penalty`、`presence_penalty`、`seed`、`stop`、`logit_bias`、`response_format`、`extra_body` を指定できます。

## ブラウザーのサポート {#browser-support}

WebGPU はデスクトップの Chrome と Edge 113 以降、Android の Chrome、Safari 26 以降で利用できます。機能を提供する前にサポート状況を確認しましょう。

```ts
import { createSSEProvider, isWebGPUSupported } from 'vue-llm/providers'

const ai = isWebGPUSupported()
  ? createWebLLM({ createEngine: CreateMLCEngine })
  : createSSEProvider({ url: '/api/chat' }) // fall back to your server
```

WebGPU が使えない場合、`validateConfig()` はわかりやすいエラーをスローし、`useChat` はそれを `error` で通知します。

::: tip メモリー
モデルには、おおよそダウンロードサイズと同程度の GPU メモリーが必要です。メモリーの少ないデバイスでは、より小さなモデルを選ぶか、`chatOptions.context_window_size` を小さくしてください。
:::
