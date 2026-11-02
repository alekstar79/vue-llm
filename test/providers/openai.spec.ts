import { afterEach, describe, expect, it, vi } from 'vitest'
import { createOpenAI } from '../../src/providers/openai'
import type { ChatStreamChunk } from '../../src/providers/types'

async function drainStream(stream: ReadableStream<ChatStreamChunk>): Promise<ChatStreamChunk[]> {
  const chunks: ChatStreamChunk[] = []
  const reader = stream.getReader()
  for (let result = await reader.read(); !result.done; result = await reader.read()) {
    chunks.push(result.value)
  }
  return chunks
}

describe('createOpenAI', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('correlates interleaved tool-call deltas by index when later chunks omit id', async () => {
    // OpenAI sends each call id once; later argument fragments are correlated by index.
    const events = [
      {
        choices: [{
          delta: {
            tool_calls: [
              { index: 0, id: 'call_1', function: { name: 'first_tool', arguments: '{' } },
              { index: 1, id: 'call_2', function: { name: 'second_tool', arguments: '{' } },
            ],
          },
          finish_reason: null,
        }],
      },
      {
        choices: [{
          delta: {
            tool_calls: [
              { index: 1, function: { arguments: '"second":true}' } },
              { index: 0, function: { arguments: '"first":true}' } },
            ],
          },
          finish_reason: null,
        }],
      },
      { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
    ]
    const body = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
    vi.stubGlobal('window', undefined)
    vi.stubGlobal('fetch', vi.fn(async () => new Response(body, { status: 200 })))

    const provider = createOpenAI({ apiKey: 'test-key' })
    const { stream } = await provider.chat({ model: 'test-model', messages: [] })
    const chunks = await drainStream(stream)

    expect(chunks).toEqual([
      { type: 'tool-call-start', toolCallId: 'call_1', toolName: 'first_tool' },
      { type: 'tool-call-delta', toolCallId: 'call_1', args: '{' },
      { type: 'tool-call-start', toolCallId: 'call_2', toolName: 'second_tool' },
      { type: 'tool-call-delta', toolCallId: 'call_2', args: '{' },
      { type: 'tool-call-delta', toolCallId: 'call_2', args: '"second":true}' },
      { type: 'tool-call-delta', toolCallId: 'call_1', args: '"first":true}' },
      { type: 'tool-call-end', toolCallId: 'call_1' },
      { type: 'tool-call-end', toolCallId: 'call_2' },
      { type: 'finish', finishReason: 'tool-calls', usage: undefined },
    ])
  })
})
