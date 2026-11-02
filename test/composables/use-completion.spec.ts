import { describe, it, expect, vi } from 'vitest'
import { useCompletion } from '../../src/composables/use-completion'
import type { LLMProvider, ChatRequest, ChatStreamChunk } from '../../src/providers/types'

function createMockProvider(chunks: ChatStreamChunk[]): LLMProvider {
  return {
    id: 'mock',
    supportsStreaming: true,
    validateConfig: vi.fn(),
    chat: vi.fn(async (_request: ChatRequest) => {
      const stream = new ReadableStream<ChatStreamChunk>({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(chunk)
          controller.close()
        },
      })
      return {
        stream,
        response: Promise.resolve({ finishReason: 'stop' as const }),
      }
    }),
  }
}

describe('useCompletion', () => {
  it('completes a prompt and returns text', async () => {
    const provider = createMockProvider([
      { type: 'text-delta', text: 'Paris' },
      { type: 'text-delta', text: ' is the capital' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { complete, completion, isLoading } = useCompletion({
      provider,
      model: 'test',
    })

    const result = await complete('What is the capital of France?')

    expect(result).toBe('Paris is the capital')
    expect(completion.value).toBe('Paris is the capital')
    expect(isLoading.value).toBe(false)
  })

  it('uses input ref when no prompt given', async () => {
    const provider = createMockProvider([
      { type: 'text-delta', text: 'answer' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { complete, input, completion } = useCompletion({ provider, model: 'test' })
    input.value = 'question'

    await complete()

    expect(completion.value).toBe('answer')
    expect(input.value).toBe('')
  })

  it('does not complete empty prompt', async () => {
    const provider = createMockProvider([])

    const { complete } = useCompletion({ provider, model: 'test' })
    await complete('')

    expect(provider.chat).not.toHaveBeenCalled()
  })

  it('passes system prompt', async () => {
    const provider = createMockProvider([
      { type: 'text-delta', text: 'ok' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { complete } = useCompletion({
      provider,
      model: 'test',
      system: 'Be concise',
    })

    await complete('Hi')

    const call = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(call.messages[0]).toMatchObject({ role: 'system', content: 'Be concise' })
  })

  it('calls onFinish with completion text', async () => {
    const onFinish = vi.fn()
    const provider = createMockProvider([
      { type: 'text-delta', text: 'done' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { complete } = useCompletion({ provider, model: 'test', onFinish })
    await complete('go')

    expect(onFinish).toHaveBeenCalledWith('done', { finishReason: 'stop', usage: undefined })
  })

  it('handles errors', async () => {
    const provider: LLMProvider = {
      id: 'err',
      supportsStreaming: true,
      validateConfig: vi.fn(),
      chat: vi.fn(async () => { throw new Error('boom') }),
    }

    const { complete, status, error } = useCompletion({ provider, model: 'test' })
    await complete('x')

    expect(status.value).toBe('error')
    expect(error.value?.message).toBe('boom')
  })

  it('stop aborts streaming', async () => {
    let controllerRef: ReadableStreamDefaultController<ChatStreamChunk> | null = null

    const provider: LLMProvider = {
      id: 'slow',
      supportsStreaming: true,
      validateConfig: vi.fn(),
      chat: vi.fn(async () => {
        const stream = new ReadableStream<ChatStreamChunk>({
          start(controller) { controllerRef = controller },
        })
        return { stream, response: Promise.resolve({ finishReason: 'stop' as const }) }
      }),
    }

    const { complete, stop, status } = useCompletion({ provider, model: 'test' })

    const promise = complete('hi')

    stop()
    if (controllerRef) (controllerRef as ReadableStreamDefaultController<ChatStreamChunk>).close()

    await promise
    expect(status.value).toBe('ready')
  })

  it('does not report an aborted completion as an error when a new one replaces it', async () => {
    let calls = 0
    const onError = vi.fn()
    const provider: LLMProvider = {
      id: 'replacement',
      supportsStreaming: true,
      validateConfig: vi.fn(),
      chat: vi.fn(async (request: ChatRequest) => {
        calls++
        if (calls === 1) {
          return {
            stream: new ReadableStream<ChatStreamChunk>({
              start(controller) {
                request.signal?.addEventListener('abort', () => {
                  controller.error(new DOMException('Aborted', 'AbortError'))
                }, { once: true })
              },
            }),
            response: Promise.resolve({ finishReason: 'stop' as const }),
          }
        }

        return createMockProvider([
          { type: 'text-delta', text: 'second answer' },
          { type: 'finish', finishReason: 'stop' },
        ]).chat({ model: 'test', messages: [] })
      }),
    }
    const { complete, completion, error } = useCompletion({ provider, model: 'test', onError })

    await Promise.all([complete('first'), complete('second')])

    expect(onError).not.toHaveBeenCalled()
    expect(error.value).toBeUndefined()
    expect(completion.value).toBe('second answer')
  })
})
