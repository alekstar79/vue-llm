import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createWebLLM, createThinkSplitter, type CreateWebLLMEngine, type WebLLMChunk, type WebLLMConfig, type WebLLMEngine } from '../../src/providers/webllm'
import type { ChatStreamChunk } from '../../src/providers/types'

async function drainStream(stream: ReadableStream<ChatStreamChunk>): Promise<ChatStreamChunk[]> {
  const chunks: ChatStreamChunk[] = []
  const reader = stream.getReader()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
  }
  return chunks
}

function textChunks(chunks: ChatStreamChunk[]): string {
  return chunks.map(c => (c.type === 'text-delta' ? c.text : '')).join('')
}

function fakeEngine(script: WebLLMChunk[] = []) {
  const requests: Record<string, unknown>[] = []
  const engine = {
    chat: {
      completions: {
        create: vi.fn(async (request: Record<string, unknown>) => {
          requests.push(request)
          return (async function* () {
            for (const chunk of script) yield chunk
          })()
        }),
      },
    },
    reload: vi.fn(async () => { }),
    interruptGenerate: vi.fn(),
    unload: vi.fn(async () => { }),
  } satisfies WebLLMEngine
  return { engine, requests }
}

const delta = (content: string): WebLLMChunk => ({ choices: [{ delta: { content } }] })

describe('createWebLLM', () => {
  const originalGPU = Object.getOwnPropertyDescriptor(navigator, 'gpu')

  beforeEach(() => {
    Object.defineProperty(navigator, 'gpu', { value: {}, configurable: true })
  })

  afterEach(() => {
    if (originalGPU) Object.defineProperty(navigator, 'gpu', originalGPU)
    else delete (navigator as { gpu?: unknown }).gpu
  })

  it('streams text, finish reason and usage', async () => {
    const { engine, requests } = fakeEngine([
      delta('Hel'),
      delta('lo'),
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } },
    ])
    const ai = createWebLLM({ createEngine: async () => engine })

    const result = await ai.chat({
      model: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
      messages: [{ role: 'user', content: 'Hi' }],
      providerOptions: { temperature: 0.2, maxTokens: 64, headers: { ignored: 'yes' } },
    })
    const chunks = await drainStream(result.stream)

    expect(textChunks(chunks)).toBe('Hello')
    expect(chunks.at(-1)).toEqual({
      type: 'finish',
      finishReason: 'stop',
      usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 },
    })
    await expect(result.response).resolves.toMatchObject({ finishReason: 'stop', usage: { totalTokens: 5 } })
    expect(requests[0]).toMatchObject({
      messages: [{ role: 'user', content: 'Hi' }],
      stream: true,
      temperature: 0.2,
      max_tokens: 64,
    })
    expect(requests[0]).not.toHaveProperty('headers')
  })

  it('creates the engine once, reports progress and reloads on model switch', async () => {
    const { engine } = fakeEngine([delta('ok')])
    const onProgress = vi.fn()
    const createEngine = vi.fn<CreateWebLLMEngine>(async (_model, config) => {
      config.initProgressCallback({ progress: 1, text: 'done', timeElapsed: 1 })
      return engine
    })
    const ai = createWebLLM({ createEngine, onProgress, chatOptions: { context_window_size: 2048 } })

    await ai.load('model-a')
    await drainStream((await ai.chat({ model: 'model-a', messages: [] })).stream)
    expect(createEngine).toHaveBeenCalledTimes(1)
    expect(createEngine.mock.calls[0][0]).toBe('model-a')
    expect(createEngine.mock.calls[0][2]).toEqual({ context_window_size: 2048 })
    expect(onProgress).toHaveBeenCalledWith({ progress: 1, text: 'done', timeElapsed: 1 })
    expect(engine.reload).not.toHaveBeenCalled()

    await drainStream((await ai.chat({ model: 'model-b', messages: [] })).stream)
    expect(createEngine).toHaveBeenCalledTimes(1)
    expect(engine.reload).toHaveBeenCalledWith('model-b', { context_window_size: 2048 })
    expect(ai.loadedModel).toBe('model-b')
  })

  it('deduplicates concurrent loads of the same model', async () => {
    const { engine } = fakeEngine()
    const createEngine = vi.fn(async () => engine)
    const ai = createWebLLM({ createEngine })

    await Promise.all([ai.load('m'), ai.load('m'), ai.load('m')])
    expect(createEngine).toHaveBeenCalledTimes(1)
  })

  it('retries engine creation after a failed load', async () => {
    const { engine } = fakeEngine()
    const createEngine = vi.fn()
      .mockRejectedValueOnce(new Error('device lost'))
      .mockResolvedValueOnce(engine)
    const ai = createWebLLM({ createEngine })

    await expect(ai.load('m')).rejects.toThrow('device lost')
    expect(ai.loadedModel).toBeUndefined()
    await ai.load('m')
    expect(ai.loadedModel).toBe('m')
  })

  it('unloads the engine', async () => {
    const { engine } = fakeEngine()
    const createEngine = vi.fn(async () => engine)
    const ai = createWebLLM({ createEngine })

    await ai.load('m')
    await ai.unload()
    expect(engine.unload).toHaveBeenCalled()
    expect(ai.loadedModel).toBeUndefined()
    await ai.load('m')
    expect(createEngine).toHaveBeenCalledTimes(2)
  })

  it('unloads after a pending load and allows loading again', async () => {
    const firstEngine = fakeEngine().engine
    const secondEngine = fakeEngine().engine
    let resolveStarted!: () => void
    let resolveFirstEngine!: (engine: WebLLMEngine) => void
    const started = new Promise<void>(resolve => { resolveStarted = resolve })
    const createEngine = vi.fn<CreateWebLLMEngine>()
      .mockImplementationOnce(async () => {
        resolveStarted()
        return await new Promise(resolve => { resolveFirstEngine = resolve })
      })
      .mockResolvedValueOnce(secondEngine)
    const ai = createWebLLM({ createEngine })

    const loading = ai.load('m')
    await started
    const unloading = ai.unload()
    resolveFirstEngine(firstEngine)
    await Promise.all([loading, unloading])

    expect(firstEngine.unload).toHaveBeenCalledOnce()
    expect(ai.loadedModel).toBeUndefined()

    await drainStream((await ai.chat({ model: 'm', messages: [] })).stream)
    expect(createEngine).toHaveBeenCalledTimes(2)
    expect(ai.loadedModel).toBe('m')
  })

  it('maps tool calls by index and passes tools through', async () => {
    const { engine, requests } = fakeEngine([
      { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', function: { name: 'weather', arguments: '{"ci' } }] } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: 'ty":"Oslo"}' } }] } }] },
      { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
    ])
    const ai = createWebLLM({ createEngine: async () => engine })

    const result = await ai.chat({
      model: 'm',
      messages: [{ role: 'user', content: 'Weather?' }],
      tools: [{ name: 'weather', description: 'Get weather', parameters: { type: 'object' } }],
    })
    const chunks = await drainStream(result.stream)

    expect(chunks).toEqual([
      { type: 'tool-call-start', toolCallId: 'call_1', toolName: 'weather' },
      { type: 'tool-call-delta', toolCallId: 'call_1', args: '{"ci' },
      { type: 'tool-call-delta', toolCallId: 'call_1', args: 'ty":"Oslo"}' },
      { type: 'tool-call-end', toolCallId: 'call_1' },
      { type: 'finish', finishReason: 'tool-calls', usage: undefined },
    ])
    expect(requests[0].tools).toEqual([
      { type: 'function', function: { name: 'weather', description: 'Get weather', parameters: { type: 'object' } } },
    ])
  })

  it('maps images and text files, flattens non-user parts', async () => {
    const { engine, requests } = fakeEngine()
    const ai = createWebLLM({ createEngine: async () => engine })

    await drainStream((await ai.chat({
      model: 'm',
      messages: [
        { role: 'system', content: [{ type: 'text', text: 'Be brief' }] },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Describe' },
            { type: 'image', url: 'data:image/png;base64,AAAA' },
            { type: 'file', url: `data:text/plain;base64,${btoa('notes')}`, mediaType: 'text/plain' },
          ],
        },
      ],
    })).stream)

    expect(requests[0].messages).toEqual([
      { role: 'system', content: 'Be brief' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
          { type: 'text', text: 'notes' },
        ],
      },
    ])
  })

  it('rejects unsupported attachments', async () => {
    const { engine } = fakeEngine()
    const ai = createWebLLM({ createEngine: async () => engine })

    await expect(ai.chat({
      model: 'm',
      messages: [{ role: 'user', content: [{ type: 'file', url: 'data:application/pdf;base64,AAAA', mediaType: 'application/pdf' }] }],
    })).rejects.toThrow('application/pdf')
  })

  it('interrupts generation on abort', async () => {
    const { engine } = fakeEngine([delta('a')])
    const ai = createWebLLM({ createEngine: async () => engine })
    const controller = new AbortController()

    const result = await ai.chat({ model: 'm', messages: [], signal: controller.signal })
    controller.abort()
    expect(engine.interruptGenerate).toHaveBeenCalled()
    await drainStream(result.stream)
  })

  it('splits <think> blocks into reasoning', async () => {
    const { engine } = fakeEngine([delta('<thi'), delta('nk>\nHmm'), delta(' ok</th'), delta('ink>\n\nAnswer')])
    const ai = createWebLLM({ createEngine: async () => engine })

    const chunks = await drainStream((await ai.chat({ model: 'm', messages: [] })).stream)
    const reasoning = chunks.map(c => (c.type === 'reasoning-delta' ? c.text : '')).join('')

    expect(reasoning).toBe('Hmm ok')
    expect(textChunks(chunks)).toBe('Answer')
  })

  it('validates WebGPU support and the engine factory', () => {
    expect(() => createWebLLM({ createEngine: undefined as never }).validateConfig()).toThrow('engine factory')

    delete (navigator as { gpu?: unknown }).gpu
    expect(() => createWebLLM({ createEngine: async () => fakeEngine().engine }).validateConfig()).toThrow('WebGPU')
  })

  describe('precision', () => {
    const F16 = 'SmolLM2-360M-Instruct-q4f16_1-MLC'
    const F32 = 'SmolLM2-360M-Instruct-q4f32_1-MLC'

    function mockAdapter(adapter: { features: string[]; vendor?: string } | null) {
      const requestAdapter = vi.fn(async () => adapter && {
        features: new Set(adapter.features),
        info: { vendor: adapter.vendor ?? 'nvidia' },
      })
      Object.defineProperty(navigator, 'gpu', { value: { requestAdapter }, configurable: true })
      return requestAdapter
    }

    async function loadedWith(model: string, config: Partial<WebLLMConfig> = {}) {
      const { engine } = fakeEngine()
      const createEngine = vi.fn<CreateWebLLMEngine>(async () => engine)
      const ai = createWebLLM({ createEngine, ...config })
      await ai.load(model)
      return { ai, engine, model: createEngine.mock.calls[0][0] }
    }

    it('keeps f16 builds on GPUs with shader-f16', async () => {
      mockAdapter({ features: ['shader-f16'] })
      expect((await loadedWith(F16)).model).toBe(F16)
    })

    it('loads the f32 build without shader-f16', async () => {
      mockAdapter({ features: [] })
      const { ai, model } = await loadedWith(F16)
      expect(model).toBe(F32)
      expect(ai.loadedModel).toBe(F32)
    })

    it('loads the f32 build on Intel GPUs even with shader-f16', async () => {
      mockAdapter({ features: ['shader-f16'], vendor: 'intel' })
      expect((await loadedWith(F16)).model).toBe(F32)
    })

    it('loads the f32 build when no adapter is available', async () => {
      mockAdapter(null)
      expect((await loadedWith(F16)).model).toBe(F32)
    })

    it('follows an explicit precision without asking the GPU', async () => {
      const requestAdapter = mockAdapter({ features: [] })
      expect((await loadedWith(F16, { precision: 'f16' })).model).toBe(F16)
      expect((await loadedWith(F32, { precision: 'f16' })).model).toBe(F16)
      expect((await loadedWith(F16, { precision: 'f32' })).model).toBe(F32)
      expect(requestAdapter).not.toHaveBeenCalled()
    })

    it('leaves f32 ids and ids without a build alone', async () => {
      const requestAdapter = mockAdapter({ features: ['shader-f16'] })
      expect((await loadedWith(F32)).model).toBe(F32)
      expect((await loadedWith('my-custom-model')).model).toBe('my-custom-model')
      expect(requestAdapter).not.toHaveBeenCalled()
    })

    it('falls back to the requested build when the other one does not exist', async () => {
      mockAdapter({ features: [] })
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => { })
      const notFound = Object.assign(new Error('Cannot find model record'), { name: 'ModelNotFoundError' })
      const { engine } = fakeEngine()
      const createEngine = vi.fn<CreateWebLLMEngine>()
        .mockRejectedValueOnce(notFound)
        .mockResolvedValueOnce(engine)
      const ai = createWebLLM({ createEngine })

      await ai.load(F16)
      expect(createEngine.mock.calls.map(call => call[0])).toEqual([F32, F16])
      expect(ai.loadedModel).toBe(F16)
      expect(warn).toHaveBeenCalledWith(expect.stringContaining('is not available'))

      // the same on a model switch, through reload
      engine.reload.mockRejectedValueOnce(notFound)
      await ai.load('gemma3-1b-it-q4f16_1-MLC')
      expect(engine.reload).toHaveBeenNthCalledWith(1, 'gemma3-1b-it-q4f32_1-MLC', undefined)
      expect(engine.reload).toHaveBeenNthCalledWith(2, 'gemma3-1b-it-q4f16_1-MLC', undefined)
      expect(ai.loadedModel).toBe('gemma3-1b-it-q4f16_1-MLC')
      warn.mockRestore()
    })

    it('does not fall back on other load errors', async () => {
      mockAdapter({ features: [] })
      const createEngine = vi.fn<CreateWebLLMEngine>().mockRejectedValue(new Error('device lost'))
      const ai = createWebLLM({ createEngine })

      await expect(ai.load(F16)).rejects.toThrow('device lost')
      expect(createEngine).toHaveBeenCalledTimes(1)
    })

    it('treats a chat with the requested id as the loaded model', async () => {
      mockAdapter({ features: [] })
      const { ai, engine } = await loadedWith(F16)
      await drainStream((await ai.chat({ model: F16, messages: [] })).stream)
      expect(engine.reload).not.toHaveBeenCalled()
    })

    it('asks the GPU once across model switches', async () => {
      const requestAdapter = mockAdapter({ features: [] })
      const { ai, engine } = await loadedWith(F16)
      await ai.load('Llama-3.2-1B-Instruct-q4f16_1-MLC')
      expect(engine.reload).toHaveBeenCalledWith('Llama-3.2-1B-Instruct-q4f32_1-MLC', undefined)
      expect(requestAdapter).toHaveBeenCalledTimes(1)
    })
  })
})

describe('createThinkSplitter', () => {
  it('passes plain text through and keeps a lone "<" that is not a tag', () => {
    const splitter = createThinkSplitter()
    const out = [...splitter.push('a < b'), ...splitter.push(' <'), ...splitter.flush()]
    expect(out.map(c => (c.type === 'text-delta' ? c.text : '')).join('')).toBe('a < b <')
  })
})
