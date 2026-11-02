import type { ChatMessage, ChatRequest, ChatResponse, ChatStreamChunk, ChatStreamResult, ContentPart, LLMProvider } from './types'
import { SSEParserStream } from '../core/sse-parser'
import { decodeBase64Text, isTextMediaType, resolveFilePart } from '../utils/file'
import { isBrowserContext } from '../utils/browser'
import { createAPIError } from '../core/errors'

export interface AnthropicConfig {
  apiKey: string
  baseURL?: string
  dangerouslyAllowBrowser?: boolean
}

export function createAnthropic(config: AnthropicConfig): LLMProvider {
  return {
    id: 'anthropic',
    supportsStreaming: true,

    validateConfig() {
      if (!config.apiKey) {
        throw new Error('[vue-llm] Anthropic API key is required. Pass it via createAnthropic({ apiKey: "..." })')
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
      const url = (config.baseURL ?? 'https://api.anthropic.com') + '/v1/messages'

      const systemMsg = request.messages.find(m => m.role === 'system')
      const nonSystemMsgs = request.messages.filter(m => m.role !== 'system')

      const body: Record<string, unknown> = {
        model: request.model,
        messages: toAnthropicMessages(nonSystemMsgs),
        max_tokens: (request.providerOptions?.maxTokens as number) ?? 4096,
        stream: true,
      }

      if (systemMsg) {
        body.system = typeof systemMsg.content === 'string'
          ? systemMsg.content
          : systemMsg.content.map(p => ({ type: 'text', text: p.type === 'text' ? p.text : '' }))
      }

      if (request.tools?.length) {
        body.tools = request.tools.map(t => ({
          name: t.name,
          description: t.description,
          input_schema: t.parameters,
        }))
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'x-api-key': config.apiKey,
          'anthropic-version': '2023-06-01',
          'Content-Type': 'application/json',
          ...(request.providerOptions?.headers as Record<string, string> | undefined),
        },
        body: JSON.stringify(body),
        signal: request.signal,
      })

      if (!response.ok) {
        throw await createAPIError('anthropic', 'Anthropic API', response)
      }

      if (!response.body) {
        throw new Error('[vue-llm] No response body from Anthropic')
      }

      let resolveResponse: (value: ChatResponse) => void
      const responsePromise = new Promise<ChatResponse>(r => { resolveResponse = r })

      let currentToolId = ''

      const stream = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new SSEParserStream())
        .pipeThrough(new TransformStream<import('../core/sse-parser').SSEEvent, ChatStreamChunk>({
          transform(event, controller) {
            let parsed: Record<string, unknown>
            try {
              parsed = JSON.parse(event.data)
            } catch {
              return
            }

            const eventType = event.event ?? parsed.type

            switch (eventType) {
              case 'content_block_start': {
                const block = parsed.content_block as { type: string; id?: string; name?: string }
                if (block?.type === 'tool_use') {
                  currentToolId = (block as { id: string }).id
                  controller.enqueue({
                    type: 'tool-call-start',
                    toolCallId: currentToolId,
                    toolName: (block as { name: string }).name,
                  })
                }
                break
              }

              case 'content_block_delta': {
                const delta = parsed.delta as { type: string; text?: string; partial_json?: string }
                if (delta?.type === 'text_delta' && delta.text) {
                  controller.enqueue({ type: 'text-delta', text: delta.text })
                }
                if (delta?.type === 'thinking_delta' && delta.text) {
                  controller.enqueue({ type: 'reasoning-delta', text: delta.text })
                }
                if (delta?.type === 'input_json_delta' && delta.partial_json) {
                  controller.enqueue({
                    type: 'tool-call-delta',
                    toolCallId: currentToolId,
                    args: delta.partial_json,
                  })
                }
                break
              }

              case 'content_block_stop': {
                if (currentToolId) {
                  controller.enqueue({ type: 'tool-call-end', toolCallId: currentToolId })
                  currentToolId = ''
                }
                break
              }

              case 'message_delta': {
                const delta = parsed.delta as { stop_reason?: string }
                const usage = parsed.usage as { output_tokens?: number } | undefined
                const reason = delta?.stop_reason === 'tool_use' ? 'tool-calls'
                  : delta?.stop_reason === 'max_tokens' ? 'length'
                  : 'stop'

                controller.enqueue({
                  type: 'finish',
                  finishReason: reason,
                  usage: usage ? { promptTokens: 0, completionTokens: usage.output_tokens ?? 0, totalTokens: usage.output_tokens ?? 0 } : undefined,
                })

                resolveResponse!({ finishReason: reason, usage: undefined })
                break
              }

              case 'error': {
                const err = parsed.error as { message?: string } | undefined
                controller.enqueue({
                  type: 'error',
                  error: new Error(`[vue-llm] Anthropic stream error: ${err?.message ?? 'unknown'}`),
                })
                break
              }
            }
          },
        }))

      return { stream, response: responsePromise }
    },
  }
}

function toAnthropicMessages(messages: ChatMessage[]): unknown[] {
  return messages.map(msg => {
    if (msg.role === 'tool') {
      return {
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: msg.toolCallId,
          content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
        }],
      }
    }

    if (msg.role === 'assistant' && msg.toolCalls?.length) {
      const content: unknown[] = []
      if (typeof msg.content === 'string' && msg.content) {
        content.push({ type: 'text', text: msg.content })
      }
      for (const tc of msg.toolCalls) {
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.name,
          input: tc.args,
        })
      }
      return { role: 'assistant', content }
    }

    return {
      role: msg.role,
      content: typeof msg.content === 'string' ? msg.content : msg.content.map(toAnthropicContentPart),
    }
  })
}

function toAnthropicContentPart(part: ContentPart): unknown {
  if (part.type === 'text') return { type: 'text', text: part.text }

  const file = resolveFilePart(part)
  const source = file.base64
    ? { type: 'base64', media_type: file.mediaType, data: file.base64 }
    : { type: 'url', url: file.url }

  if (file.mediaType.startsWith('image/')) {
    return { type: 'image', source }
  }

  if (file.mediaType === 'application/pdf') {
    return { type: 'document', source, ...(file.filename ? { title: file.filename } : {}) }
  }

  if (isTextMediaType(file.mediaType) && file.base64) {
    return {
      type: 'document',
      source: { type: 'text', media_type: 'text/plain', data: decodeBase64Text(file.base64) },
      ...(file.filename ? { title: file.filename } : {}),
    }
  }

  throw new Error(`[vue-llm] Anthropic does not support ${file.mediaType} attachments`)
}
