import type { ChatMessage, ChatRequest, ChatResponse, ChatStreamChunk, ChatStreamResult, ContentPart, LLMProvider } from './types'
import { SSEParserStream } from '../core/sse-parser'
import { decodeBase64Text, isTextMediaType, resolveFilePart } from '../utils/file'
import { isBrowserContext } from '../utils/browser'
import { createAPIError } from '../core/errors'

export interface OpenAIConfig {
  apiKey: string
  baseURL?: string
  dangerouslyAllowBrowser?: boolean
}

export function createOpenAI(config: OpenAIConfig): LLMProvider {
  return {
    id: 'openai',
    supportsStreaming: true,

    validateConfig() {
      if (!config.apiKey) {
        throw new Error('[vue-llm] OpenAI API key is required. Pass it via createOpenAI({ apiKey: "..." })')
      }
      // API keys require explicit opt-in in both window and Worker browser contexts.
      if (!config.dangerouslyAllowBrowser && isBrowserContext()) {
        throw new Error(
          '[vue-llm] API keys must not be used in the browser. '
          + 'Use a server proxy (/api/chat) to keep keys safe. '
          + 'If you understand the risk, set dangerouslyAllowBrowser: true.',
        )
      }
    },

    async chat(request: ChatRequest): Promise<ChatStreamResult> {
      const url = config.baseURL ?? 'https://api.openai.com/v1/chat/completions'

      const body: Record<string, unknown> = {
        model: request.model,
        messages: toOpenAIMessages(request.messages),
        stream: true,
        stream_options: { include_usage: true },
      }

      if (request.providerOptions?.response_format) {
        body.response_format = request.providerOptions.response_format
      }

      if (request.tools?.length) {
        body.tools = request.tools.map(t => ({
          type: 'function',
          function: {
            name: t.name,
            description: t.description,
            parameters: t.parameters,
          },
        }))
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${config.apiKey}`,
          'Content-Type': 'application/json',
          ...(request.providerOptions?.headers as Record<string, string> | undefined),
        },
        body: JSON.stringify(body),
        signal: request.signal,
      })

      if (!response.ok) {
        throw await createAPIError('openai', 'OpenAI API', response)
      }

      if (!response.body) {
        throw new Error('[vue-llm] No response body from OpenAI')
      }

      let resolveResponse: (value: ChatResponse) => void
      const responsePromise = new Promise<ChatResponse>(r => { resolveResponse = r })

      const stream = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new SSEParserStream())
        .pipeThrough(new TransformStream<import('../core/sse-parser').SSEEvent, ChatStreamChunk>({
          transform(event, controller) {
            if (event.data === '[DONE]') return

            let parsed: OpenAIChatChunk
            try {
              parsed = JSON.parse(event.data)
            } catch {
              return
            }

            const choice = parsed.choices?.[0]
            if (!choice) {
              if (parsed.usage) {
                resolveResponse!({
                  finishReason: 'stop',
                  usage: {
                    promptTokens: parsed.usage.prompt_tokens,
                    completionTokens: parsed.usage.completion_tokens,
                    totalTokens: parsed.usage.total_tokens,
                  },
                })
              }
              return
            }

            const delta = choice.delta

            if (delta.content) {
              controller.enqueue({ type: 'text-delta', text: delta.content })
            }

            if (delta.tool_calls) {
              for (const tc of delta.tool_calls) {
                if (tc.id) {
                  controller.enqueue({
                    type: 'tool-call-start',
                    toolCallId: tc.id,
                    toolName: tc.function?.name ?? '',
                  })
                }
                if (tc.function?.arguments) {
                  controller.enqueue({
                    type: 'tool-call-delta',
                    toolCallId: tc.id ?? '',
                    args: tc.function.arguments,
                  })
                }
              }
            }

            if (choice.finish_reason) {
              const pendingToolCalls = delta.tool_calls?.filter(tc => tc.id) ?? []
              for (const tc of pendingToolCalls) {
                if (tc.id) {
                  controller.enqueue({ type: 'tool-call-end', toolCallId: tc.id })
                }
              }

              const reason = choice.finish_reason === 'tool_calls' ? 'tool-calls'
                : choice.finish_reason === 'length' ? 'length'
                : 'stop'

              controller.enqueue({ type: 'finish', finishReason: reason, usage: undefined })
            }
          },
        }))

      return { stream, response: responsePromise }
    },
  }
}

function toOpenAIMessages(messages: ChatMessage[]): unknown[] {
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
        content: typeof msg.content === 'string' && msg.content ? msg.content : null,
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

    return {
      role: msg.role,
      content: typeof msg.content === 'string' ? msg.content : msg.content.map(toOpenAIContentPart),
    }
  })
}

function toOpenAIContentPart(part: ContentPart): unknown {
  if (part.type === 'text') return { type: 'text', text: part.text }

  const file = resolveFilePart(part)

  if (file.mediaType.startsWith('image/')) {
    return { type: 'image_url', image_url: { url: file.url } }
  }

  if (!file.base64) {
    throw new Error(`[vue-llm] OpenAI requires ${file.mediaType} attachments as data URLs, got a remote URL`)
  }

  if (file.mediaType === 'application/pdf') {
    return {
      type: 'file',
      file: { filename: file.filename ?? 'document.pdf', file_data: file.url },
    }
  }

  const audioFormat = OPENAI_AUDIO_FORMATS[file.mediaType]
  if (audioFormat) {
    return { type: 'input_audio', input_audio: { data: file.base64, format: audioFormat } }
  }

  if (isTextMediaType(file.mediaType)) {
    return { type: 'text', text: decodeBase64Text(file.base64) }
  }

  throw new Error(`[vue-llm] OpenAI does not support ${file.mediaType} attachments`)
}

const OPENAI_AUDIO_FORMATS: Record<string, string> = {
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
  'audio/mpeg': 'mp3',
  'audio/mp3': 'mp3',
}

interface OpenAIChatChunk {
  choices?: {
    delta: {
      content?: string
      tool_calls?: {
        id?: string
        function?: { name?: string; arguments?: string }
      }[]
    }
    finish_reason?: string
  }[]
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
}
