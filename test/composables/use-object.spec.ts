import { describe, it, expect, vi } from 'vitest'
import { useObject } from '../../src/composables/use-object'
import type { LLMProvider, ChatRequest, ChatStreamChunk, ChatStreamResult } from '../../src/providers/types'

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

interface TestSchema {
  name: string
  age: number
  city: string
}

const schema = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    age: { type: 'number' },
    city: { type: 'string' },
  },
  required: ['name', 'age', 'city'],
}

describe('useObject', () => {
  it('parses complete JSON response', async () => {
    const json = JSON.stringify({ name: 'Alice', age: 30, city: 'London' })
    const provider = createMockProvider([
      { type: 'text-delta', text: json },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { submit, object } = useObject<TestSchema>({ provider, model: 'test', schema })

    const result = await submit('Extract info')

    expect(result).toEqual({ name: 'Alice', age: 30, city: 'London' })
    expect(object.value).toEqual({ name: 'Alice', age: 30, city: 'London' })
  })

  it('builds partial object from streaming chunks', async () => {
    const provider = createMockProvider([
      { type: 'text-delta', text: '{"name": "Bob"' },
      { type: 'text-delta', text: ', "age": 25' },
      { type: 'text-delta', text: ', "city": "Paris"}' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { submit, object, partialObject } = useObject<TestSchema>({ provider, model: 'test', schema })

    await submit('Extract')

    expect(object.value).toEqual({ name: 'Bob', age: 25, city: 'Paris' })
    expect(partialObject.value).toEqual({ name: 'Bob', age: 25, city: 'Paris' })
  })

  it('includes schema in system prompt', async () => {
    const provider = createMockProvider([
      { type: 'text-delta', text: '{"name":"X","age":1,"city":"Y"}' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { submit } = useObject<TestSchema>({ provider, model: 'test', schema })
    await submit('go')

    const call = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(call.messages[0].role).toBe('system')
    expect(call.messages[0].content).toContain('"type": "object"')
  })

  it('calls onFinish with parsed object', async () => {
    const onFinish = vi.fn()
    const provider = createMockProvider([
      { type: 'text-delta', text: '{"name":"A","age":1,"city":"B"}' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { submit } = useObject<TestSchema>({ provider, model: 'test', schema, onFinish })
    await submit('go')

    expect(onFinish).toHaveBeenCalledWith(
      { name: 'A', age: 1, city: 'B' },
      { finishReason: 'stop', usage: undefined },
    )
  })

  it('errors on invalid JSON', async () => {
    const provider = createMockProvider([
      { type: 'text-delta', text: 'not json at all' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const { submit, status, error } = useObject<TestSchema>({ provider, model: 'test', schema })
    await submit('go')

    expect(status.value).toBe('error')
    expect(error.value?.message).toContain('Failed to parse JSON')
  })

  it('does not submit empty prompt', async () => {
    const provider = createMockProvider([])

    const { submit } = useObject<TestSchema>({ provider, model: 'test', schema })
    await submit('')

    expect(provider.chat).not.toHaveBeenCalled()
  })

  it('treats stop during a pending provider call as cancellation', async () => {
    let resolveChat!: (result: ChatStreamResult) => void
    const provider: LLMProvider = {
      id: 'pending',
      supportsStreaming: true,
      validateConfig: vi.fn(),
      chat: vi.fn(() => new Promise<ChatStreamResult>(resolve => { resolveChat = resolve })),
    }
    const object = useObject<TestSchema>({ provider, model: 'test', schema })
    const pending = object.submit('extract an object')

    await vi.waitFor(() => expect(resolveChat).toBeTypeOf('function'))
    object.stop()
    resolveChat({
      stream: new ReadableStream<ChatStreamChunk>({ start(controller) { controller.close() } }),
      response: Promise.resolve({ finishReason: 'stop' }),
    })

    await expect(pending).resolves.toBeUndefined()
    expect(object.status.value).toBe('ready')
    expect(object.error.value).toBeUndefined()
  })
})
