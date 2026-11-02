import type { ChatMessage, ChatRequest, ChatResponse, ChatStreamChunk, ChatStreamResult, LLMProvider } from './types'
import { resolveFilePart } from '../utils/file'
import { isBrowserContext } from '../utils/browser'
import { createAPIError } from '../core/errors'

export interface GoogleConfig {
  apiKey: string
  baseURL?: string
  dangerouslyAllowBrowser?: boolean
}

export function createGoogle(config: GoogleConfig): LLMProvider {
  return {
    id: 'google',
    supportsStreaming: true,

    validateConfig() {
      if (!config.apiKey) {
        throw new Error('[vue-llm] Google AI API key is required. Pass it via createGoogle({ apiKey: "..." })')
      }
      // API keys require explicit opt-in in both window and Worker browser contexts.
      if (!config.dangerouslyAllowBrowser && isBrowserContext()) {
        throw new Error(
          '[vue-llm] API keys must not be used in the browser. '
          + 'Use a server proxy to keep keys safe. '
          + 'If you understand the risk, set dangerouslyAllowBrowser: true.',
        )
      }
    },

    async chat(request: ChatRequest): Promise<ChatStreamResult> {
      const baseURL = config.baseURL ?? 'https://generativelanguage.googleapis.com/v1beta'
      const url = `${baseURL}/models/${request.model}:streamGenerateContent?alt=sse&key=${config.apiKey}`

      const body: GeminiRequest = {
        contents: toGeminiContents(request.messages),
      }

      const systemMsg = request.messages.find(m => m.role === 'system')
      if (systemMsg) {
        body.systemInstruction = {
          parts: [{ text: typeof systemMsg.content === 'string' ? systemMsg.content : '' }],
        }
      }

      if (request.tools?.length) {
        body.tools = [{
          functionDeclarations: request.tools.map(t => ({
            name: t.name,
            description: t.description,
            parameters: t.parameters,
          })),
        }]
      }

      body.generationConfig = {
        ...(request.providerOptions?.response_format as Record<string, unknown>)?.type === 'json_object'
          ? { responseMimeType: 'application/json' }
          : {},
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(request.providerOptions?.headers as Record<string, string> | undefined),
        },
        body: JSON.stringify(body),
        signal: request.signal,
      })

      if (!response.ok) {
        throw await createAPIError('google', 'Google AI API', response)
      }

      if (!response.body) {
        throw new Error('[vue-llm] No response body from Google AI')
      }

      let resolveResponse: (value: ChatResponse) => void
      const responsePromise = new Promise<ChatResponse>(r => { resolveResponse = r })

      let toolCallCounter = 0
      let sawFunctionCall = false

      const stream = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream<string, string>({
          buffer: '',
          transform(chunk, controller) {
            (this as unknown as { buffer: string }).buffer += chunk
            const lines = (this as unknown as { buffer: string }).buffer.split('\n')
            ;(this as unknown as { buffer: string }).buffer = lines.pop()!
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                controller.enqueue(line.slice(6))
              }
            }
          },
          flush(controller) {
            const remaining = (this as unknown as { buffer: string }).buffer.trim()
            if (remaining.startsWith('data: ')) {
              controller.enqueue(remaining.slice(6))
            }
          },
        } as Transformer<string, string>))
        .pipeThrough(new TransformStream<string, ChatStreamChunk>({
          transform(data, controller) {
            let parsed: GeminiStreamChunk
            try {
              parsed = JSON.parse(data)
            } catch {
              return
            }

            const candidate = parsed.candidates?.[0]
            if (!candidate) return

            const parts = candidate.content?.parts
            if (parts) {
              for (const part of parts) {
                if (part.text !== undefined) {
                  controller.enqueue({ type: 'text-delta', text: part.text })
                }

                if (part.functionCall) {
                  const id = `tc_gemini_${++toolCallCounter}`
                  sawFunctionCall = true

                  controller.enqueue({
                    type: 'tool-call-start',
                    toolCallId: id,
                    toolName: part.functionCall.name,
                  })
                  controller.enqueue({
                    type: 'tool-call-delta',
                    toolCallId: id,
                    args: JSON.stringify(part.functionCall.args ?? {}),
                  })
                  controller.enqueue({ type: 'tool-call-end', toolCallId: id })
                }
              }
            }

            if (candidate.finishReason) {
              const reason = sawFunctionCall ? 'tool-calls'
                : candidate.finishReason === 'MAX_TOKENS' ? 'length'
                : 'stop'

              const usage = parsed.usageMetadata
              controller.enqueue({
                type: 'finish',
                finishReason: reason,
                usage: usage ? {
                  promptTokens: usage.promptTokenCount ?? 0,
                  completionTokens: usage.candidatesTokenCount ?? 0,
                  totalTokens: usage.totalTokenCount ?? 0,
                } : undefined,
              })

              resolveResponse!({
                finishReason: reason,
                usage: usage ? {
                  promptTokens: usage.promptTokenCount ?? 0,
                  completionTokens: usage.candidatesTokenCount ?? 0,
                  totalTokens: usage.totalTokenCount ?? 0,
                } : undefined,
              })
            }
          },
        }))

      return { stream, response: responsePromise }
    },
  }
}

function toGeminiContents(messages: ChatMessage[]): GeminiContent[] {
  const contents: GeminiContent[] = []
  const toolNames = new Map<string, string>()

  for (const msg of messages) {
    if (msg.role === 'system') continue

    if (msg.role === 'tool') {
      const part: GeminiPart = {
        functionResponse: {
          name: toolNames.get(msg.toolCallId ?? '') ?? msg.toolCallId ?? '',
          response: { result: parseToolResult(msg.content) },
        },
      }
      const last = contents.at(-1)
      if (last?.role === 'user' && last.parts.every(p => p.functionResponse)) {
        last.parts.push(part)
      } else {
        contents.push({ role: 'user', parts: [part] })
      }
      continue
    }

    for (const tc of msg.toolCalls ?? []) {
      toolNames.set(tc.id, tc.name)
    }

    const role = msg.role === 'assistant' ? 'model' : 'user'
    const parts: GeminiPart[] = []

    if (typeof msg.content === 'string') {
      if (msg.content) parts.push({ text: msg.content })
    } else if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === 'text') parts.push({ text: part.text })
        if (part.type === 'image' || part.type === 'file') {
          const file = resolveFilePart(part)
          parts.push(file.base64
            ? { inlineData: { mimeType: file.mediaType, data: file.base64 } }
            : { fileData: { mimeType: file.mediaType, fileUri: file.url } })
        }
      }
    }

    if (msg.toolCalls?.length) {
      for (const tc of msg.toolCalls) {
        parts.push({
          functionCall: {
            name: tc.name,
            args: typeof tc.args === 'string' ? JSON.parse(tc.args) : tc.args,
          },
        })
      }
    }

    if (parts.length > 0) {
      contents.push({ role, parts })
    }
  }

  return contents
}

function parseToolResult(content: ChatMessage['content']): unknown {
  if (typeof content !== 'string') return content
  try {
    return JSON.parse(content)
  } catch {
    return content
  }
}

interface GeminiRequest {
  contents: GeminiContent[]
  systemInstruction?: { parts: { text: string }[] }
  tools?: { functionDeclarations: { name: string; description: string; parameters: Record<string, unknown> }[] }[]
  generationConfig?: Record<string, unknown>
}

interface GeminiContent {
  role: string
  parts: GeminiPart[]
}

interface GeminiPart {
  text?: string
  functionCall?: { name: string; args?: unknown }
  functionResponse?: { name: string; response: unknown }
  inlineData?: { mimeType: string; data: string }
  fileData?: { mimeType: string; fileUri: string }
}

interface GeminiStreamChunk {
  candidates?: {
    content?: { parts: GeminiPart[] }
    finishReason?: string
  }[]
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
  }
}
