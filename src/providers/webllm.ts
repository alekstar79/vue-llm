import type { ChatMessage, ChatRequest, ChatResponse, ChatStreamChunk, ChatStreamResult, ContentPart, LLMProvider } from './types'
import type { FinishReason, Usage } from '../types'
import { decodeBase64Text, isTextMediaType, resolveFilePart } from '../utils/file'

export interface WebLLMProgress {
  /** 0..1 */
  progress: number
  text: string
  timeElapsed: number
}

/** The subset of `@mlc-ai/web-llm`'s `MLCEngine` that vue-llm uses. */
export interface WebLLMEngine {
  chat: {
    completions: {
      create(request: Record<string, unknown>): Promise<AsyncIterable<WebLLMChunk>>
    }
  }
  reload(model: string, chatOptions?: Record<string, unknown>): Promise<void>
  interruptGenerate(): void | Promise<void>
  unload(): Promise<void>
}

export type CreateWebLLMEngine = (
  model: string,
  engineConfig: { initProgressCallback: (report: WebLLMProgress) => void } & Record<string, unknown>,
  chatOptions?: Record<string, unknown>,
) => Promise<WebLLMEngine>

export interface WebLLMConfig {
  /**
   * Engine factory, usually `CreateMLCEngine` from `@mlc-ai/web-llm`,
   * or a wrapper around `CreateWebWorkerMLCEngine` to run the model off the main thread.
   */
  createEngine: CreateWebLLMEngine
  /** Extra `MLCEngineConfig` (e.g. a custom `appConfig`) */
  engineConfig?: Record<string, unknown>
  /** Overrides for the model's `mlc-chat-config.json` (e.g. `context_window_size`) */
  chatOptions?: Record<string, unknown>
  /** Download / compile progress while a model is loading */
  onProgress?: (report: WebLLMProgress) => void
  /**
   * Which build of a model to load. Most models ship 16-bit (`q4f16`) and 32-bit (`q4f32`) builds.
   * - `'auto'` (default): swaps a 16-bit id for its 32-bit build when the GPU can't run f16 reliably,
   *   see {@link detectWebGPUPrecision}
   * - `'f16'` / `'f32'`: always load that build
   */
  precision?: WebLLMPrecision | 'auto'
}

export type WebLLMPrecision = 'f16' | 'f32'

export interface WebLLMProvider extends LLMProvider {
  /** Download and compile a model ahead of the first message */
  load(model: string): Promise<void>
  /** Free GPU memory. The next request loads the model again. */
  unload(): Promise<void>
  /** Model currently on the GPU, if any. Differs from the requested id when `precision` swapped the build. */
  readonly loadedModel: string | undefined
}

/** True when the browser exposes WebGPU, which in-browser models require. */
export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && !!(navigator as { gpu?: unknown }).gpu
}

interface GPUAdapterLike {
  features: { has(feature: string): boolean }
  info?: { vendor?: string }
}

/**
 * The model build this GPU runs reliably: `'f32'` when there is no adapter, no `shader-f16` feature,
 * or the GPU is Intel, whose drivers report `shader-f16` but return gibberish from f16 models.
 */
export async function detectWebGPUPrecision(): Promise<WebLLMPrecision> {
  try {
    const gpu = (navigator as { gpu?: { requestAdapter(): Promise<GPUAdapterLike | null> } }).gpu
    const adapter = await gpu?.requestAdapter()
    if (!adapter?.features.has('shader-f16')) return 'f32'
    return adapter.info?.vendor?.toLowerCase().includes('intel') ? 'f32' : 'f16'
  } catch {
    return 'f32'
  }
}

// MLC ids encode the build as `-q4f16_1-`, `-q0f32-` and so on
const BUILD_IN_ID = /-q(\d)f(16|32)(_\d+)?-/

export function createWebLLM(config: WebLLMConfig): WebLLMProvider {
  let enginePromise: Promise<WebLLMEngine> | undefined
  // The id passed to load() and the id on the GPU; they differ when the build was swapped
  let requestedModel: string | undefined
  let loadedModel: string | undefined
  let loading: Promise<void> | undefined
  let detected: Promise<WebLLMPrecision> | undefined

  const onProgress = (report: WebLLMProgress): void => config.onProgress?.(report)

  async function resolveModel(model: string): Promise<string> {
    const build = BUILD_IN_ID.exec(model)
    if (!build) return model

    const setting = config.precision ?? 'auto'
    // auto only ever moves f16 to f32; an f32 id was picked on purpose
    if (setting === 'auto' && build[2] === '32') return model

    const precision = setting === 'auto' ? await (detected ??= detectWebGPUPrecision()) : setting
    return model.replace(BUILD_IN_ID, `-q${build[1]}${precision}${build[3] ?? ''}-`)
  }

  async function loadEngine(model: string): Promise<void> {
    if (!enginePromise) {
      enginePromise = config.createEngine(
        model,
        { ...config.engineConfig, initProgressCallback: onProgress },
        config.chatOptions,
      )
      try {
        await enginePromise
      } catch (error) {
        enginePromise = undefined
        throw error
      }
    } else {
      await (await enginePromise).reload(model, config.chatOptions)
    }
  }

  function serialize(operation: () => Promise<void>): Promise<void> {
    const previous = loading ?? Promise.resolve()
    const next = previous.catch(() => { }).then(operation)
    loading = next
    const clear = (): void => {
      if (loading === next) loading = undefined
    }
    next.then(clear, clear)
    return next
  }

  function load(model: string): Promise<void> {
    if (requestedModel === model && !loading) return Promise.resolve()

    return serialize(async () => {
      if (requestedModel === model) return
      requestedModel = undefined
      loadedModel = undefined

      let resolved = await resolveModel(model)
      try {
        await loadEngine(resolved)
      } catch (error) {
        // Not every model has the other build (gemma3-1b-it has no q4f32), so fall back to the id as given
        if (resolved === model || (error as Error | undefined)?.name !== 'ModelNotFoundError') throw error
        console.warn(`[vue-llm] ${resolved} is not available, loading ${model} instead`)
        resolved = model
        await loadEngine(resolved)
      }
      requestedModel = model
      loadedModel = resolved
    })
  }

  return {
    id: 'webllm',
    supportsStreaming: true,

    get loadedModel() {
      return loadedModel
    },

    validateConfig() {
      if (typeof config.createEngine !== 'function') {
        throw new Error(
          '[vue-llm] createWebLLM requires an engine factory. '
          + 'Pass createWebLLM({ createEngine: CreateMLCEngine }) from @mlc-ai/web-llm',
        )
      }
      if (!isWebGPUSupported()) {
        throw new Error('[vue-llm] In-browser models require WebGPU, which this browser does not support')
      }
    },

    load,

    unload() {
      return serialize(async () => {
        const engine = enginePromise
        enginePromise = undefined
        requestedModel = undefined
        loadedModel = undefined
        if (engine) await (await engine).unload()
      })
    },

    async chat(request: ChatRequest): Promise<ChatStreamResult> {
      await load(request.model)
      const engine = await enginePromise!

      if (request.signal?.aborted) {
        throw request.signal.reason ?? new DOMException('Aborted', 'AbortError')
      }

      const body: Record<string, unknown> = {
        ...pickGenerationOptions(request.providerOptions),
        messages: toWebLLMMessages(request.messages),
        stream: true,
        stream_options: { include_usage: true },
      }

      if (request.providerOptions?.response_format) {
        body.response_format = request.providerOptions.response_format
      }

      if (request.tools?.length) {
        body.tools = request.tools.map(t => ({
          type: 'function',
          function: { name: t.name, description: t.description, parameters: t.parameters },
        }))
      }

      const iterable = await engine.chat.completions.create(body)
      const onAbort = (): void => {
        void engine.interruptGenerate()
      }
      request.signal?.addEventListener('abort', onAbort, { once: true })

      let resolveResponse: (value: ChatResponse) => void
      let rejectResponse: (reason: unknown) => void
      const response = new Promise<ChatResponse>((resolve, reject) => {
        resolveResponse = resolve
        rejectResponse = reject
      })
      // Consumers may only read the stream; don't surface an unhandled rejection
      response.catch(() => { })

      const iterator = iterable[Symbol.asyncIterator]()
      const think = createThinkSplitter()
      const toolIds = new Map<number, string>()
      let finishReason: FinishReason = 'stop'
      let usage: Usage | undefined

      const cleanup = (): void => request.signal?.removeEventListener('abort', onAbort)

      const mapChunk = (value: WebLLMChunk): ChatStreamChunk[] => {
        if (value.usage) {
          usage = {
            promptTokens: value.usage.prompt_tokens,
            completionTokens: value.usage.completion_tokens,
            totalTokens: value.usage.total_tokens,
          }
        }

        const choice = value.choices?.[0]
        if (!choice) return []
        const out: ChatStreamChunk[] = []

        if (choice.delta?.content) {
          out.push(...think.push(choice.delta.content))
        }

        for (const [position, tc] of (choice.delta?.tool_calls ?? []).entries()) {
          const index = tc.index ?? position
          let id = toolIds.get(index)
          if (!id) {
            id = tc.id ?? `webllm-call-${index}-${Date.now().toString(36)}`
            toolIds.set(index, id)
            out.push({ type: 'tool-call-start', toolCallId: id, toolName: tc.function?.name ?? '' })
          }
          if (tc.function?.arguments) {
            const args = typeof tc.function.arguments === 'string'
              ? tc.function.arguments
              : JSON.stringify(tc.function.arguments)
            out.push({ type: 'tool-call-delta', toolCallId: id, args })
          }
        }

        if (choice.finish_reason) {
          finishReason = choice.finish_reason === 'tool_calls' ? 'tool-calls'
            : choice.finish_reason === 'length' ? 'length'
            : 'stop'
        }

        return out
      }

      const stream = new ReadableStream<ChatStreamChunk>({
        async pull(controller) {
          try {
            // A pull must enqueue something or the stream stalls, so skip chunks that map to nothing
            while (true) {
              const { done, value } = await iterator.next()

              if (done) {
                for (const chunk of think.flush()) controller.enqueue(chunk)
                for (const id of toolIds.values()) controller.enqueue({ type: 'tool-call-end', toolCallId: id })
                toolIds.clear()
                controller.enqueue({ type: 'finish', finishReason, usage })
                controller.close()
                cleanup()
                resolveResponse!({ finishReason, usage })
                return
              }

              const out = mapChunk(value)
              for (const chunk of out) controller.enqueue(chunk)
              if (out.length) return
            }
          } catch (error) {
            cleanup()
            rejectResponse!(error)
            controller.error(error)
          }
        },

        async cancel() {
          cleanup()
          await engine.interruptGenerate()
          await iterator.return?.()
          resolveResponse!({ finishReason: 'stop', usage })
        },
      })

      return { stream, response }
    },
  }
}

const GENERATION_OPTIONS = [
  'temperature',
  'top_p',
  'max_tokens',
  'frequency_penalty',
  'presence_penalty',
  'seed',
  'stop',
  'logit_bias',
  'extra_body',
] as const

function pickGenerationOptions(options: Record<string, unknown> | undefined): Record<string, unknown> {
  const picked: Record<string, unknown> = {}
  if (!options) return picked
  for (const key of GENERATION_OPTIONS) {
    if (options[key] !== undefined) picked[key] = options[key]
  }
  if (options.maxTokens !== undefined && picked.max_tokens === undefined) picked.max_tokens = options.maxTokens
  return picked
}

function toWebLLMMessages(messages: ChatMessage[]): unknown[] {
  return messages.map(msg => {
    if (msg.role === 'tool') {
      return {
        role: 'tool',
        content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
        tool_call_id: msg.toolCallId,
      }
    }

    if (msg.role === 'assistant' && msg.toolCalls?.length) {
      return {
        role: 'assistant',
        content: typeof msg.content === 'string' ? msg.content : '',
        tool_calls: msg.toolCalls.map(tc => ({
          id: tc.id,
          type: 'function',
          function: {
            name: tc.name,
            arguments: typeof tc.args === 'string' ? tc.args : JSON.stringify(tc.args),
          },
        })),
      }
    }

    if (typeof msg.content === 'string') {
      return { role: msg.role, content: msg.content }
    }

    // Only user messages may carry image parts; flatten everything else to text
    const parts = msg.content.map(toWebLLMContentPart)
    if (msg.role !== 'user' || parts.every(p => p.type === 'text')) {
      return { role: msg.role, content: parts.map(p => (p.type === 'text' ? p.text : '')).join('\n') }
    }
    return { role: msg.role, content: parts }
  })
}

type WebLLMContentPart = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }

function toWebLLMContentPart(part: ContentPart): WebLLMContentPart {
  if (part.type === 'text') return { type: 'text', text: part.text }

  const file = resolveFilePart(part)

  if (file.mediaType.startsWith('image/')) {
    return { type: 'image_url', image_url: { url: file.url } }
  }

  if (file.base64 && isTextMediaType(file.mediaType)) {
    return { type: 'text', text: decodeBase64Text(file.base64) }
  }

  throw new Error(`[vue-llm] In-browser models do not support ${file.mediaType} attachments`)
}

const THINK_OPEN = '<think>'
const THINK_CLOSE = '</think>'

/**
 * Splits `<think>…</think>` blocks emitted inline by reasoning models (Qwen3, DeepSeek-R1 distills)
 * into reasoning deltas. Tags may be split across chunks, so a possible tag prefix is held back.
 */
export function createThinkSplitter(): {
  push(text: string): ChatStreamChunk[]
  flush(): ChatStreamChunk[]
} {
  let buffer = ''
  let inThink = false
  // Models put newlines right after a tag; drop them even when they arrive in a later chunk
  let trimLeading = false

  const emit = (text: string, out: ChatStreamChunk[]): void => {
    if (!text) return
    out.push(inThink ? { type: 'reasoning-delta', text } : { type: 'text-delta', text })
  }

  return {
    push(text) {
      const out: ChatStreamChunk[] = []
      buffer += text

      while (buffer) {
        if (trimLeading) {
          buffer = buffer.replace(/^\s+/, '')
          if (!buffer) break
          trimLeading = false
        }

        const tag = inThink ? THINK_CLOSE : THINK_OPEN
        const index = buffer.indexOf(tag)

        if (index !== -1) {
          emit(buffer.slice(0, index), out)
          buffer = buffer.slice(index + tag.length)
          inThink = !inThink
          trimLeading = true
          continue
        }

        const keep = partialTagLength(buffer, tag)
        emit(buffer.slice(0, buffer.length - keep), out)
        buffer = buffer.slice(buffer.length - keep)
        break
      }

      return out
    },

    flush() {
      const out: ChatStreamChunk[] = []
      emit(buffer, out)
      buffer = ''
      return out
    },
  }
}

function partialTagLength(text: string, tag: string): number {
  for (let length = Math.min(tag.length - 1, text.length); length > 0; length--) {
    if (tag.startsWith(text.slice(-length))) return length
  }
  return 0
}

export interface WebLLMChunk {
  choices?: {
    delta?: {
      content?: string | null
      tool_calls?: {
        index?: number
        id?: string
        function?: { name?: string; arguments?: string | Record<string, unknown> }
      }[]
    }
    finish_reason?: string | null
  }[]
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
}
