import { describe, it, expect, vi, afterEach } from 'vitest'
import { LLMAPIError, parseRetryAfter } from '../src/core/errors'
import { rateLimitMiddleware, retryMiddleware, wrapProvider } from '../src/middleware'
import { createLLMHandler } from '../src/server'
import { createOpenAI } from '../src/providers/openai'
import type { ChatRequest, ChatStreamChunk, ChatStreamResult, LLMProvider } from '../src/providers/types'

function okResult(): ChatStreamResult {
  return {
    stream: new ReadableStream<ChatStreamChunk>({
      start(controller) {
        controller.enqueue({ type: 'finish', finishReason: 'stop' })
        controller.close()
      },
    }),
    response: Promise.resolve({ finishReason: 'stop' }),
  }
}

function apiError(status: number, retryAfter?: number): LLMAPIError {
  return new LLMAPIError({ message: `error ${status}`, status, provider: 'mock', body: '', retryAfter })
}

function mockProvider(chat: (req: ChatRequest) => Promise<ChatStreamResult>): LLMProvider {
  return { id: 'mock', supportsStreaming: true, validateConfig: () => {}, chat: vi.fn(chat) }
}

const request: ChatRequest = { model: 'm', messages: [{ role: 'user', content: 'hi' }] }

async function drain(stream: ReadableStream<unknown>): Promise<void> {
  const reader = stream.getReader()
  while (!(await reader.read()).done) { /* consume */ }
}

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('LLMAPIError', () => {
  it('is thrown by providers with status and Retry-After', async () => {
    vi.stubGlobal('window', undefined)
    vi.stubGlobal('fetch', vi.fn(async () => new Response('slow down', {
      status: 429,
      headers: { 'retry-after': '2' },
    })))

    const error = await createOpenAI({ apiKey: 'k' }).chat(request).catch(e => e)

    expect(error).toBeInstanceOf(LLMAPIError)
    expect(error).toMatchObject({ status: 429, provider: 'openai', body: 'slow down', retryAfter: 2000, isRetryable: true })
    expect(error.message).toBe('[vue-llm] OpenAI API error 429: slow down')
  })

  it('parses retry-after-ms, seconds and HTTP dates', () => {
    expect(parseRetryAfter(new Headers({ 'retry-after-ms': '150' }))).toBe(150)
    expect(parseRetryAfter(new Headers({ 'retry-after': '3' }))).toBe(3000)
    const date = new Date(Date.now() + 5000).toUTCString()
    expect(parseRetryAfter(new Headers({ 'retry-after': date }))).toBeGreaterThan(3000)
    expect(parseRetryAfter(new Headers())).toBeUndefined()
  })

  it('classifies retryable statuses', () => {
    expect(apiError(503).isRetryable).toBe(true)
    expect(apiError(400).isRetryable).toBe(false)
    expect(apiError(401).isRetryable).toBe(false)
  })
})

describe('retryMiddleware', () => {
  it('actually retries the request until it succeeds', async () => {
    let calls = 0
    const provider = mockProvider(async () => {
      calls++
      if (calls < 3) throw apiError(503)
      return okResult()
    })
    const onRetry = vi.fn()
    const wrapped = wrapProvider(provider, [retryMiddleware({ initialDelay: 1, onRetry })])

    await wrapped.chat(request)

    expect(provider.chat).toHaveBeenCalledTimes(3)
    expect(onRetry).toHaveBeenCalledTimes(2)
    expect(onRetry.mock.calls.map(c => c[0].attempt)).toEqual([1, 2])
  })

  it('uses exponential backoff without jitter', async () => {
    vi.useFakeTimers()
    const provider = mockProvider(async () => { throw apiError(500) })
    const delays: number[] = []
    const wrapped = wrapProvider(provider, [retryMiddleware({
      maxRetries: 3,
      initialDelay: 100,
      jitter: false,
      onRetry: ({ delay }) => delays.push(delay),
    })])

    const promise = wrapped.chat(request).catch(e => e)
    await vi.runAllTimersAsync()
    const error = await promise

    expect(delays).toEqual([100, 200, 400])
    expect(error.status).toBe(500)
    expect(provider.chat).toHaveBeenCalledTimes(4)
  })

  it('honors retryAfter, capped by maxDelay', async () => {
    const delays: number[] = []
    let calls = 0
    const provider = mockProvider(async () => {
      if (calls++ === 0) throw apiError(429, 50_000)
      return okResult()
    })
    vi.useFakeTimers()
    const wrapped = wrapProvider(provider, [retryMiddleware({ maxDelay: 10, onRetry: ({ delay }) => delays.push(delay) })])

    const promise = wrapped.chat(request)
    await vi.runAllTimersAsync()
    await promise

    expect(delays).toEqual([10])
  })

  it('does not retry non-retryable errors', async () => {
    const provider = mockProvider(async () => { throw apiError(401) })
    const wrapped = wrapProvider(provider, [retryMiddleware({ initialDelay: 1 })])

    await expect(wrapped.chat(request)).rejects.toMatchObject({ status: 401 })
    expect(provider.chat).toHaveBeenCalledTimes(1)
  })

  it('stops waiting when the request is aborted', async () => {
    const provider = mockProvider(async () => { throw apiError(503) })
    const controller = new AbortController()
    const wrapped = wrapProvider(provider, [retryMiddleware({ initialDelay: 60_000 })])

    const promise = wrapped.chat({ ...request, signal: controller.signal })
    controller.abort()

    await expect(promise).rejects.toThrow()
    expect(provider.chat).toHaveBeenCalledTimes(1)
  })

  it('keeps the legacy (maxRetries, delay) signature', async () => {
    const provider = mockProvider(async () => { throw apiError(502) })
    const wrapped = wrapProvider(provider, [retryMiddleware(2, 1)])

    await expect(wrapped.chat(request)).rejects.toMatchObject({ status: 502 })
    expect(provider.chat).toHaveBeenCalledTimes(3)
  })
})

describe('rateLimitMiddleware', () => {
  it('limits concurrent requests until streams finish', async () => {
    const provider = mockProvider(async () => okResult())
    const wrapped = wrapProvider(provider, [rateLimitMiddleware({ maxConcurrent: 1 })])

    const first = await wrapped.chat(request)
    let secondStarted = false
    const second = wrapped.chat(request).then(r => { secondStarted = true; return r })

    await new Promise(r => setTimeout(r, 10))
    expect(secondStarted).toBe(false)

    await drain(first.stream)
    await drain((await second).stream)
    expect(secondStarted).toBe(true)
  })

  it('limits requests per interval', async () => {
    vi.useFakeTimers()
    const provider = mockProvider(async () => okResult())
    const wrapped = wrapProvider(provider, [rateLimitMiddleware({ maxRequests: 2, interval: 1000 })])

    await wrapped.chat(request)
    await wrapped.chat(request)
    const third = wrapped.chat(request)

    await vi.advanceTimersByTimeAsync(500)
    expect(provider.chat).toHaveBeenCalledTimes(2)

    await vi.advanceTimersByTimeAsync(600)
    await third
    expect(provider.chat).toHaveBeenCalledTimes(3)
  })

  it('rejects queued requests on abort', async () => {
    const provider = mockProvider(async () => okResult())
    const wrapped = wrapProvider(provider, [rateLimitMiddleware({ maxConcurrent: 1 })])
    await wrapped.chat(request)

    const controller = new AbortController()
    const queued = wrapped.chat({ ...request, signal: controller.signal })
    controller.abort()

    await expect(queued).rejects.toThrow()
    expect(provider.chat).toHaveBeenCalledTimes(1)
  })
})

describe('createLLMHandler rateLimit', () => {
  it('returns 429 with Retry-After per client key', async () => {
    const provider = mockProvider(async () => okResult())
    const handler = createLLMHandler({ provider, rateLimit: { maxRequests: 1, interval: 30_000 } })
    const post = (ip: string) => handler(new Request('http://x/api/chat', {
      method: 'POST',
      headers: { 'x-forwarded-for': `${ip}, 10.0.0.1` },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    }))

    expect((await post('1.1.1.1')).status).toBe(200)

    const limited = await post('1.1.1.1')
    expect(limited.status).toBe(429)
    expect(limited.headers.get('Retry-After')).toBe('30')

    expect((await post('2.2.2.2')).status).toBe(200)
  })

  it('uses the configured identity key instead of a caller-controlled forwarding header', async () => {
    const provider = mockProvider(async () => okResult())
    const handler = createLLMHandler({
      provider,
      rateLimit: {
        maxRequests: 1,
        interval: 30_000,
        key: request => request.headers.get('x-user-id') ?? 'anonymous',
      },
    })
    const post = (userId: string, ip: string) => handler(new Request('http://x/api/chat', {
      method: 'POST',
      headers: { 'x-user-id': userId, 'x-forwarded-for': ip },
      body: JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }),
    }))

    expect((await post('user-a', '1.1.1.1')).status).toBe(200)
    expect((await post('user-a', '2.2.2.2')).status).toBe(429)
    expect((await post('user-b', '1.1.1.1')).status).toBe(200)
  })
})
