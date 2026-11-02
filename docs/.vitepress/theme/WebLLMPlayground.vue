<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { useChat } from '../../../src/composables/use-chat'
import { createWebLLM, detectWebGPUPrecision, isWebGPUSupported, type WebLLMProvider } from '../../../src/providers/webllm'
import { renderMarkdown } from '../../../src/core/markdown'
import type { UIMessage } from '../../../src/types'
import { track } from './analytics'

// Each model ships 16-bit (q4f16) and 32-bit (q4f32) builds. f16 is smaller and faster but needs the
// WebGPU `shader-f16` feature; without it, or with buggy drivers (Intel), f16 models answer with gibberish.
const MODELS = [
  { base: 'SmolLM2-360M-Instruct', label: 'SmolLM2 360M', size: { f16: '~380 MB', f32: '~580 MB' }, note: 'fastest download' },
  { base: 'Qwen2.5-0.5B-Instruct', label: 'Qwen2.5 0.5B', size: { f16: '~950 MB', f32: '~1.1 GB' }, note: 'good all-rounder' },
  { base: 'Llama-3.2-1B-Instruct', label: 'Llama 3.2 1B', size: { f16: '~880 MB', f32: '~1.1 GB' }, note: 'best answers' },
  { base: 'Qwen3-0.6B', label: 'Qwen3 0.6B', size: { f16: '~1.4 GB', f32: '~1.9 GB' }, note: 'shows its reasoning' },
]

const supported = isWebGPUSupported()
const baseModel = ref(MODELS[0].base)
// Unknown until the adapter is checked; 32-bit is the safe default
const f16Supported = ref(false)
const force32 = ref(false)
const precision = computed<'f16' | 'f32'>(() => (f16Supported.value && !force32.value ? 'f16' : 'f32'))
const model = computed(() => `${baseModel.value}-q4${precision.value}_1-MLC`)
const loadState = ref<'idle' | 'loading' | 'ready' | 'error'>('idle')
const progress = ref(0)
const progressText = ref('')
const loadError = ref('')
const stats = ref('')
const provider = shallowRef<WebLLMProvider>()
const scroller = ref<HTMLElement>()

let worker: Worker | undefined
let startedAt = 0

function getProvider(): WebLLMProvider {
  if (provider.value) return provider.value

  provider.value = createWebLLM({
    async createEngine(modelId, engineConfig, chatOptions) {
      const { CreateWebWorkerMLCEngine } = await import('@mlc-ai/web-llm')
      worker = new Worker(new URL('./webllm.worker.ts', import.meta.url), { type: 'module' })
      return CreateWebWorkerMLCEngine(worker, modelId, engineConfig, chatOptions)
    },
    onProgress(report) {
      progress.value = report.progress
      progressText.value = report.text
    },
  })
  return provider.value
}

// useChat validates the provider before every request, so hand it a lazy proxy
const lazyProvider: WebLLMProvider = {
  id: 'webllm',
  supportsStreaming: true,
  get loadedModel() {
    return provider.value?.loadedModel
  },
  validateConfig: () => getProvider().validateConfig(),
  load: m => getProvider().load(m),
  unload: () => getProvider().unload(),
  chat: request => getProvider().chat(request),
}

const { messages, input, status, error, sendMessage, stop, setMessages } = useChat({
  provider: lazyProvider,
  model,
  system: 'You are a friendly assistant running entirely in the user\'s browser. Answer concisely.',
  onFinish(_message, { usage }) {
    if (!usage || !startedAt) return
    const seconds = (performance.now() - startedAt) / 1000
    stats.value = `${usage.completionTokens} tokens · ${(usage.completionTokens / seconds).toFixed(1)} tok/s`
  },
})

const busy = computed(() => status.value === 'submitted' || status.value === 'streaming')
const selected = computed(() => MODELS.find(m => m.base === baseModel.value)!)

async function load(): Promise<void> {
  loadState.value = 'loading'
  loadError.value = ''
  progress.value = 0
  const started = performance.now()
  const seconds = () => Math.round((performance.now() - started) / 1000)
  try {
    await getProvider().load(model.value)
    loadState.value = 'ready'
    track('playground_model_loaded', { model: model.value, precision: precision.value, seconds: seconds() })
  } catch (e) {
    loadState.value = 'error'
    loadError.value = e instanceof Error ? e.message : String(e)
    track('playground_model_failed', { model: model.value, precision: precision.value, seconds: seconds(), error: loadError.value.slice(0, 120) })
  }
}

watch(model, () => {
  if (loadState.value !== 'idle') {
    stop()
    setMessages([])
    void load()
  }
})

function submit(): void {
  if (!input.value.trim() || busy.value || loadState.value !== 'ready') return
  stats.value = ''
  startedAt = performance.now()
  track('playground_message', { model: model.value })
  void sendMessage()
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    submit()
  }
}

function partsOf(message: UIMessage, type: 'text' | 'reasoning'): string {
  return message.parts.map(p => (p.type === type ? p.text : '')).join('')
}

watch(
  () => messages.value.map(m => m.parts.length + partsOf(m, 'text').length + partsOf(m, 'reasoning').length).join(),
  async () => {
    await nextTick()
    scroller.value?.scrollTo({ top: scroller.value.scrollHeight })
  },
)

const suggestions = ['Explain Vue reactivity in two sentences', 'Write a haiku about TypeScript', 'What is WebGPU?']

onMounted(async () => {
  if (!supported) track('playground_unsupported', { userAgent: navigator.userAgent.slice(0, 120) })
  else f16Supported.value = (await detectWebGPUPrecision()) === 'f16'
})

onBeforeUnmount(() => {
  stop()
  worker?.terminate()
})
</script>

<template>
  <div class="playground">
    <div v-if="!supported" class="notice">
      Your browser does not support <strong>WebGPU</strong>, which in-browser models need.
      Try a recent Chrome, Edge or Safari 26+ on a desktop.
    </div>

    <template v-else>
      <div class="toolbar">
        <select v-model="baseModel" :disabled="loadState === 'loading' || busy" aria-label="Model">
          <option v-for="m in MODELS" :key="m.base" :value="m.base">
            {{ m.label }} · {{ m.size[precision] }} · {{ m.note }}
          </option>
        </select>
        <button v-if="loadState === 'idle' || loadState === 'error'" class="primary" @click="load">
          Load model ({{ selected.size[precision] }})
        </button>
        <span v-else-if="loadState === 'ready'" class="badge">Running on your GPU</span>
      </div>

      <label class="precision">
        <input v-model="force32" type="checkbox" :disabled="!f16Supported || loadState === 'loading' || busy">
        <span v-if="f16Supported">32-bit mode — turn on if answers look like gibberish (larger download)</span>
        <span v-else>Using 32-bit models: 16-bit ones don't run reliably on this GPU</span>
      </label>

      <div v-if="loadState === 'loading'" class="progress">
        <!-- WebLLM reports progress per finished shard, so animate until the first one lands -->
        <div :class="['bar', { indeterminate: progress === 0 }]">
          <div :style="{ width: `${Math.round(progress * 100)}%` }" />
        </div>
        <small>{{ progress === 0 ? 'Downloading model files… the first visit takes a minute, later visits load from cache.' : progressText }}</small>
      </div>

      <div v-if="loadError" class="notice error">
        {{ loadError }}
      </div>

      <div ref="scroller" class="messages">
        <p v-if="!messages.length" class="empty">
          <template v-if="loadState === 'ready'">
            The model is ready. Nothing you type leaves this tab.
          </template>
          <template v-else>
            Load a model to chat with it — it is downloaded once, cached by the browser, and runs locally on WebGPU.
            No API key, no server.
          </template>
        </p>

        <div v-for="message in messages" :key="message.id" :class="['message', message.role]">
          <details v-if="partsOf(message, 'reasoning')" class="reasoning" :open="status === 'streaming' && !partsOf(message, 'text')">
            <summary>Reasoning</summary>
            <div>{{ partsOf(message, 'reasoning') }}</div>
          </details>
          <div v-if="message.role === 'user'" class="bubble">
            {{ partsOf(message, 'text') }}
          </div>
          <!-- renderMarkdown escapes all model output, so v-html is safe here -->
          <div v-else class="bubble markdown" v-html="renderMarkdown(partsOf(message, 'text'))" />
        </div>

        <div v-if="status === 'submitted'" class="message assistant">
          <div class="bubble typing">
            …
          </div>
        </div>
      </div>

      <div v-if="loadState === 'ready' && !messages.length" class="suggestions">
        <button v-for="s in suggestions" :key="s" @click="input = s; submit()">
          {{ s }}
        </button>
      </div>

      <form class="composer" @submit.prevent="submit">
        <textarea
          v-model="input"
          rows="2"
          :disabled="loadState !== 'ready'"
          :placeholder="loadState === 'ready' ? 'Ask anything… (Enter to send)' : 'Load a model first'"
          @keydown="onKeydown"
        />
        <button v-if="busy" type="button" @click="stop">
          Stop
        </button>
        <button v-else type="submit" class="primary" :disabled="loadState !== 'ready' || !input.trim()">
          Send
        </button>
      </form>

      <div class="footer">
        <span v-if="error" class="error-text">{{ error.message }}</span>
        <span v-else>{{ stats }}</span>
        <button v-if="messages.length && !busy" class="link" @click="setMessages([])">
          Clear
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.playground {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 16px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

select,
textarea {
  font: inherit;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 6px 10px;
}

select {
  flex: 1 1 260px;
  min-width: 0;
}

button {
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  border-radius: 8px;
  padding: 6px 14px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
  transition: border-color 0.2s;
}

button:hover:not(:disabled) {
  border-color: var(--vp-c-brand-1);
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

button.primary {
  background: var(--vp-c-brand-3);
  border-color: var(--vp-c-brand-3);
  color: var(--vp-c-white);
}

button.primary:hover:not(:disabled) {
  background: var(--vp-c-brand-2);
}

button.link {
  border: none;
  background: none;
  padding: 0;
  color: var(--vp-c-brand-1);
}

.precision {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--vp-c-text-2);
  cursor: pointer;
}

.precision input:disabled {
  cursor: default;
}

.badge {
  font-size: 13px;
  color: var(--vp-c-green-1);
  background: var(--vp-c-green-soft);
  border-radius: 999px;
  padding: 2px 10px;
}

.progress {
  margin-top: 12px;
}

.bar {
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  overflow: hidden;
}

.bar > div {
  height: 100%;
  background: var(--vp-c-brand-1);
  transition: width 0.3s;
}

.bar.indeterminate > div {
  width: 30% !important;
  animation: slide 1.4s ease-in-out infinite;
}

@keyframes slide {
  from { transform: translateX(-100%); }
  to { transform: translateX(340%); }
}

@media (prefers-reduced-motion: reduce) {
  .bar.indeterminate > div {
    animation: none;
  }
}

.progress small {
  display: block;
  margin-top: 6px;
  color: var(--vp-c-text-2);
  overflow-wrap: anywhere;
}

.messages {
  margin-top: 12px;
  height: 360px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px;
}

.empty {
  margin: auto;
  max-width: 420px;
  text-align: center;
  color: var(--vp-c-text-2);
  font-size: 14px;
}

.message {
  display: flex;
  flex-direction: column;
  max-width: 85%;
}

.message.user {
  align-self: flex-end;
  align-items: flex-end;
}

.bubble {
  padding: 8px 12px;
  border-radius: 12px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.user .bubble {
  background: var(--vp-c-brand-soft);
  border-color: transparent;
}

.bubble.markdown {
  white-space: normal;
}

.bubble.markdown :deep(p),
.bubble.markdown :deep(ul),
.bubble.markdown :deep(ol),
.bubble.markdown :deep(pre) {
  margin: 0 0 8px;
}

.bubble.markdown :deep(> :last-child) {
  margin-bottom: 0;
}

.bubble.markdown :deep(pre) {
  overflow-x: auto;
  padding: 8px;
  border-radius: 6px;
  background: var(--vp-c-bg-alt);
}

.typing {
  color: var(--vp-c-text-2);
}

.reasoning {
  font-size: 13px;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
  white-space: pre-wrap;
}

.reasoning summary {
  cursor: pointer;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.suggestions button {
  font-size: 13px;
  border-radius: 999px;
}

.composer {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  align-items: flex-end;
}

.composer textarea {
  flex: 1;
  min-width: 0;
  resize: vertical;
}

.footer {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  min-height: 20px;
  margin-top: 6px;
  font-size: 12px;
  color: var(--vp-c-text-2);
}

.notice {
  padding: 12px;
  border-radius: 8px;
  background: var(--vp-c-warning-soft);
  font-size: 14px;
}

.notice.error {
  margin-top: 12px;
  background: var(--vp-c-danger-soft);
  overflow-wrap: anywhere;
}

.error-text {
  color: var(--vp-c-danger-1);
}
</style>
