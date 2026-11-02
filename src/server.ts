import type { LLMProvider, ChatMessage, ChatStreamChunk } from './providers/types'
import type { ToolDefinition } from './types'
import { createSSEProvider } from './providers/sse'

export interface ServerToolDefinition extends ToolDefinition {
  execute?: (args: unknown, context: { request: Request; toolCallId: string }) => unknown | Promise<unknown>
}

export interface LLMHandlerOptions {
  provider: LLMProvider
  model?: string
  system?: string
  tools?: ServerToolDefinition[]
  maxToolRoundtrips?: number
  maxMessages?: number
  rateLimit?: ServerRateLimitOptions
  onRequest?: (body: LLMRequestBody) => LLMRequestBody | Promise<LLMRequestBody>
}

export interface ServerRateLimitOptions {
  maxRequests: number
  interval?: number
  /**
   * Choose an authenticated identity or an address resolved by a trusted proxy.
   * Example: `req => req.headers.get('x-user-id') ?? 'anonymous'`.
   */
  key?: (request: Request) => string
}

export interface LLMRequestBody {
  messages: ChatMessage[]
  model?: string
  system?: string
}

export function createLLMHandler(options: LLMHandlerOptions) {
  const limiter = options.rateLimit ? createWindowLimiter(options.rateLimit) : undefined

  return async (request: Request): Promise<Response> => {
    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (limiter) {
      const retryAfter = limiter(request)
      if (retryAfter !== undefined) {
        return new Response(JSON.stringify({ error: 'Too many requests' }), {
          status: 429,
          headers: {
            'Content-Type': 'application/json',
            'Retry-After': String(Math.ceil(retryAfter / 1000)),
          },
        })
      }
    }

    let parsedBody: unknown
    try {
      parsedBody = await request.json()
    } catch {
      return new Response(JSON.stringify({ error: 'Invalid JSON body' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // Validate the runtime JSON shape before treating messages as ChatMessage[].
    const requestBody = parsedBody as Record<string, unknown> | null
    if (
      !requestBody
      || typeof requestBody !== 'object'
      || Array.isArray(requestBody)
      || !Array.isArray(requestBody.messages)
      || requestBody.messages.length === 0
    ) {
      return new Response(JSON.stringify({ error: 'messages array is required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    let body = parsedBody as LLMRequestBody
    if (options.maxMessages && body.messages.length > options.maxMessages) {
      body.messages = body.messages.slice(-options.maxMessages)
    }

    try {
      // Keep user hook failures inside the handler's HTTP error boundary.
      if (options.onRequest) {
        body = await options.onRequest(body)
      }

      const messages: ChatMessage[] = []
      const systemText = body.system ?? options.system
      if (systemText) {
        messages.push({ role: 'system', content: systemText })
      }
      messages.push(...body.messages)

      options.provider.validateConfig()

      const chatRequest = {
        model: body.model ?? options.model ?? 'gpt-4o',
        tools: options.tools?.map(({ execute: _execute, ...tool }) => tool),
        signal: request.signal,
      }

      const { stream } = await options.provider.chat({ ...chatRequest, messages })

      const executable = new Map(
        (options.tools ?? []).filter(tool => tool.execute).map(tool => [tool.name, tool]),
      )

      const sseStream = streamToSSE(executable.size > 0
        ? runServerTools(stream, {
          messages,
          executable,
          maxRoundtrips: options.maxToolRoundtrips ?? 5,
          request,
          next: history => options.provider.chat({ ...chatRequest, messages: history }).then(r => r.stream),
        })
        : stream)

      return new Response(sseStream, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      return new Response(JSON.stringify({ error: message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      })
    }
  }
}

interface ServerToolRun {
  messages: ChatMessage[]
  executable: Map<string, ServerToolDefinition>
  maxRoundtrips: number
  request: Request
  next: (messages: ChatMessage[]) => Promise<ReadableStream<ChatStreamChunk>>
}

function runServerTools(first: ReadableStream<ChatStreamChunk>, run: ServerToolRun): ReadableStream<ChatStreamChunk> {
  let reader: ReadableStreamDefaultReader<ChatStreamChunk> | undefined

  return new ReadableStream<ChatStreamChunk>({
    async start(controller) {
      const history = [...run.messages]
      let current = first

      try {
        for (let round = 0; ; round++) {
          let text = ''
          const calls = new Map<string, { name: string; args: string }>()
          let finish: Extract<ChatStreamChunk, { type: 'finish' }> | undefined

          reader = current.getReader()
          for (let r = await reader.read(); !r.done; r = await reader.read()) {
            const chunk = r.value
            if (chunk.type === 'finish') {
              finish = chunk
              continue
            }
            if (chunk.type === 'text-delta') text += chunk.text
            if (chunk.type === 'tool-call-start') calls.set(chunk.toolCallId, { name: chunk.toolName, args: '' })
            if (chunk.type === 'tool-call-delta') {
              const call = calls.get(chunk.toolCallId)
              if (call) call.args += chunk.args
            }
            controller.enqueue(chunk)
          }

          const toolCalls = [...calls].map(([id, call]) => ({ id, name: call.name, args: parseArgs(call.args) }))
          const runnable = toolCalls.filter(tc => run.executable.has(tc.name))

          if (finish?.finishReason !== 'tool-calls' || runnable.length === 0 || round >= run.maxRoundtrips) {
            if (finish) controller.enqueue(finish)
            break
          }

          history.push({ role: 'assistant', content: text, toolCalls })

          for (const tc of runnable) {
            try {
              const result = await run.executable.get(tc.name)!.execute!(tc.args, { request: run.request, toolCallId: tc.id })
              controller.enqueue({ type: 'tool-result', toolCallId: tc.id, result })
              history.push({ role: 'tool', toolCallId: tc.id, content: JSON.stringify(result ?? null) })
            } catch (err) {
              const error = err instanceof Error ? err.message : String(err)
              controller.enqueue({ type: 'tool-result', toolCallId: tc.id, error })
              history.push({ role: 'tool', toolCallId: tc.id, content: JSON.stringify({ error }) })
            }
          }

          if (runnable.length < toolCalls.length) {
            controller.enqueue(finish)
            break
          }

          current = await run.next(history)
        }
      } catch (err) {
        if (run.request.signal.aborted) return
        controller.enqueue({ type: 'error', error: err instanceof Error ? err : new Error(String(err)) })
      }
      controller.close()
    },
    cancel(reason) {
      return reader?.cancel(reason)
    },
  })
}

function parseArgs(args: string): unknown {
  if (!args) return {}
  try {
    return JSON.parse(args)
  } catch {
    return args
  }
}

function createWindowLimiter(options: ServerRateLimitOptions): (request: Request) => number | undefined {
  const interval = options.interval ?? 60_000
  const getKey = options.key ?? defaultRateLimitKey
  const windows = new Map<string, { start: number; count: number }>()

  return (request) => {
    const now = Date.now()
    const key = getKey(request)

    if (windows.size > 10_000) {
      for (const [k, w] of windows) {
        if (now - w.start >= interval) windows.delete(k)
      }
    }

    let window = windows.get(key)
    if (!window || now - window.start >= interval) {
      window = { start: now, count: 0 }
      windows.set(key, window)
    }

    if (window.count >= options.maxRequests) {
      return window.start + interval - now
    }

    window.count++
    return undefined
  }
}

function defaultRateLimitKey(request: Request): string {
  // Keep the forwarding-header default for compatibility; use key() unless a trusted proxy overwrites it.
  // Example: key: req => req.headers.get('x-trusted-client-ip') ?? 'anonymous'
  const forwarded = request.headers.get('x-forwarded-for')
  return forwarded?.split(',')[0].trim()
    || request.headers.get('x-real-ip')
    || 'global'
}

export function streamToSSE(stream: ReadableStream<ChatStreamChunk>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()

  return stream.pipeThrough(new TransformStream<ChatStreamChunk, Uint8Array>({
    transform(chunk, controller) {
      const data = serializeChunk(chunk)
      if (data) {
        controller.enqueue(encoder.encode(`data: ${data}\n\n`))
      }
    },
    flush(controller) {
      controller.enqueue(encoder.encode('data: [DONE]\n\n'))
    },
  }))
}

function serializeChunk(chunk: ChatStreamChunk): string | null {
  if (chunk.type === 'error') {
    return JSON.stringify({ type: 'error', message: chunk.error.message })
  }
  return JSON.stringify(chunk)
}

export function createProxyProvider(baseURL: string): LLMProvider {
  return createSSEProvider({ id: 'proxy', name: 'Proxy', url: baseURL })
}
