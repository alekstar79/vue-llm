import { describe, it, expect, vi } from 'vitest'
import { createLLMHandler, streamToSSE, createProxyProvider } from '../src/server'
import type { LLMProvider, ChatRequest, ChatStreamChunk } from '../src/providers/types'

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

function makeRequest(body: unknown, method = 'POST'): Request {
  return new Request('http://localhost/api/chat', {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'POST' ? JSON.stringify(body) : undefined,
  })
}

async function readSSE(response: Response): Promise<string[]> {
  const text = await response.text()
  return text.split('\n\n').filter(l => l.startsWith('data: ')).map(l => l.slice(6))
}

describe('createLLMHandler', () => {
  it('streams SSE response from provider', async () => {
    const provider = createMockProvider([
      { type: 'text-delta', text: 'Hello' },
      { type: 'text-delta', text: ' world' },
      { type: 'finish', finishReason: 'stop' },
    ])

    const handler = createLLMHandler({ provider, model: 'test' })
    const req = makeRequest({ messages: [{ role: 'user', content: 'hi' }] })
    const res = await handler(req)

    expect(res.status).toBe(200)
    expect(res.headers.get('Content-Type')).toBe('text/event-stream')

    const events = await readSSE(res)
    expect(events).toContain(JSON.stringify({ type: 'text-delta', text: 'Hello' }))
    expect(events).toContain(JSON.stringify({ type: 'text-delta', text: ' world' }))
    expect(events).toContain('[DONE]')
  })

  it('rejects non-POST methods', async () => {
    const provider = createMockProvider([])
    const handler = createLLMHandler({ provider })
    const req = new Request('http://localhost/api/chat', { method: 'GET' })
    const res = await handler(req)

    expect(res.status).toBe(405)
  })

  it('rejects empty messages', async () => {
    const provider = createMockProvider([])
    const handler = createLLMHandler({ provider })
    const req = makeRequest({ messages: [] })
    const res = await handler(req)

    expect(res.status).toBe(400)
    const body = await res.json()
    expect(body.error).toContain('messages')
  })

  it('rejects a non-array messages value without calling the provider', async () => {
    const provider = createMockProvider([])
    const handler = createLLMHandler({ provider })
    const res = await handler(makeRequest({ messages: 'not-an-array' }))

    expect(res.status).toBe(400)
    expect(provider.chat).not.toHaveBeenCalled()
  })

  it('prepends system message from options', async () => {
    const provider = createMockProvider([
      { type: 'finish', finishReason: 'stop' },
    ])

    const handler = createLLMHandler({ provider, model: 'test', system: 'Be helpful' })
    const req = makeRequest({ messages: [{ role: 'user', content: 'hi' }] })
    await handler(req)

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(chatCall.messages[0]).toEqual({ role: 'system', content: 'Be helpful' })
    expect(chatCall.messages[1]).toEqual({ role: 'user', content: 'hi' })
  })

  it('uses model from request body over options', async () => {
    const provider = createMockProvider([
      { type: 'finish', finishReason: 'stop' },
    ])

    const handler = createLLMHandler({ provider, model: 'default-model' })
    const req = makeRequest({
      messages: [{ role: 'user', content: 'hi' }],
      model: 'custom-model',
    })
    await handler(req)

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(chatCall.model).toBe('custom-model')
  })

  it('limits messages with maxMessages', async () => {
    const provider = createMockProvider([
      { type: 'finish', finishReason: 'stop' },
    ])

    const handler = createLLMHandler({ provider, model: 'test', maxMessages: 2 })
    const req = makeRequest({
      messages: [
        { role: 'user', content: 'first' },
        { role: 'assistant', content: 'reply' },
        { role: 'user', content: 'second' },
      ],
    })
    await handler(req)

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0][0]
    const userMessages = chatCall.messages.filter((m: { role: string }) => m.role !== 'system')
    expect(userMessages).toHaveLength(2)
    expect(userMessages[0].content).toBe('reply')
    expect(userMessages[1].content).toBe('second')
  })

  it('calls onRequest hook', async () => {
    const provider = createMockProvider([
      { type: 'finish', finishReason: 'stop' },
    ])

    const handler = createLLMHandler({
      provider,
      model: 'test',
      onRequest: (body) => ({ ...body, messages: [{ role: 'user' as const, content: 'intercepted' }] }),
    })
    const req = makeRequest({ messages: [{ role: 'user', content: 'original' }] })
    await handler(req)

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(chatCall.messages[0].content).toBe('intercepted')
  })

  it('returns an HTTP error when onRequest throws', async () => {
    const provider = createMockProvider([])
    const handler = createLLMHandler({
      provider,
      onRequest: () => { throw new Error('invalid request') },
    })
    const res = await handler(makeRequest({ messages: [{ role: 'user', content: 'hi' }] }))

    expect(res.status).toBe(500)
    expect(await res.json()).toMatchObject({ error: 'invalid request' })
    expect(provider.chat).not.toHaveBeenCalled()
  })

  it('returns 500 on provider error', async () => {
    const provider: LLMProvider = {
      id: 'mock',
      supportsStreaming: true,
      validateConfig: vi.fn(),
      chat: vi.fn(async () => { throw new Error('Provider failed') }),
    }

    const handler = createLLMHandler({ provider, model: 'test' })
    const req = makeRequest({ messages: [{ role: 'user', content: 'hi' }] })
    const res = await handler(req)

    expect(res.status).toBe(500)
    const body = await res.json()
    expect(body.error).toContain('Provider failed')
  })
})

describe('streamToSSE', () => {
  it('converts ChatStreamChunk to SSE format', async () => {
    const chunks: ChatStreamChunk[] = [
      { type: 'text-delta', text: 'hi' },
      { type: 'finish', finishReason: 'stop' },
    ]

    const stream = new ReadableStream<ChatStreamChunk>({
      start(controller) {
        for (const c of chunks) controller.enqueue(c)
        controller.close()
      },
    })

    const sseStream = streamToSSE(stream)
    const reader = sseStream.getReader()
    const decoder = new TextDecoder()
    let result = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      result += decoder.decode(value, { stream: true })
    }

    expect(result).toContain('data: {"type":"text-delta","text":"hi"}')
    expect(result).toContain('data: [DONE]')
  })
})

describe('createProxyProvider', () => {
  it('creates a provider with id proxy', () => {
    const provider = createProxyProvider('/api/chat')
    expect(provider.id).toBe('proxy')
    expect(provider.supportsStreaming).toBe(true)
  })

  it('validateConfig does not throw', () => {
    const provider = createProxyProvider('/api/chat')
    expect(() => provider.validateConfig()).not.toThrow()
  })
})
