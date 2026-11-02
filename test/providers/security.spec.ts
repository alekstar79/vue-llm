import { afterEach, describe, expect, it, vi } from 'vitest'
import { createAnthropic } from '../../src/providers/anthropic'
import { createGoogle } from '../../src/providers/google'
import { createOpenAI } from '../../src/providers/openai'
import type { LLMProvider } from '../../src/providers/types'

const keyedProviders: [string, () => LLMProvider][] = [
  ['OpenAI', () => createOpenAI({ apiKey: 'test-key' })],
  ['Anthropic', () => createAnthropic({ apiKey: 'test-key' })],
  ['Google', () => createGoogle({ apiKey: 'test-key' })],
]

afterEach(() => vi.unstubAllGlobals())

describe('API-key browser guard', () => {
  it.each(keyedProviders)('%s requires explicit opt-in inside a Worker', (_name, create) => {
    // Workers expose importScripts but do not have window, so they still need the browser opt-in.
    vi.stubGlobal('window', undefined)
    vi.stubGlobal('importScripts', vi.fn())

    expect(() => create().validateConfig()).toThrow('API keys must not be used in the browser')
  })

  it.each([
    ['OpenAI', () => createOpenAI({ apiKey: 'test-key', dangerouslyAllowBrowser: true })],
    ['Anthropic', () => createAnthropic({ apiKey: 'test-key', dangerouslyAllowBrowser: true })],
    ['Google', () => createGoogle({ apiKey: 'test-key', dangerouslyAllowBrowser: true })],
  ] as [string, () => LLMProvider][])('%s keeps the explicit Worker opt-in', (_name, create) => {
    vi.stubGlobal('window', undefined)
    vi.stubGlobal('importScripts', vi.fn())

    expect(() => create().validateConfig()).not.toThrow()
  })

  it('does not treat a server runtime as a browser Worker', () => {
    vi.stubGlobal('window', undefined)
    vi.stubGlobal('importScripts', undefined)

    expect(() => createOpenAI({ apiKey: 'test-key' }).validateConfig()).not.toThrow()
  })
})
