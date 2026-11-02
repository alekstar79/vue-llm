import { computed, getCurrentScope, onScopeDispose, ref, type Ref } from 'vue'
import { type MaybeRefOrGetter, toValue } from '../utils/vue'
import type { ChatStatus, FinishReason, Usage } from '../types'
import type { LLMProvider } from '../providers/types'
import { generateId } from '../utils/generate-id'
import { registerDevtoolsSession, tracedChat } from '../core/devtools'

export interface UseCompletionOptions {
  id?: string
  provider: LLMProvider
  model: MaybeRefOrGetter<string>
  system?: MaybeRefOrGetter<string | undefined>
  onFinish?: (completion: string, context: { finishReason: FinishReason; usage?: Usage }) => void
  onError?: (error: Error) => void
}

export interface UseCompletionReturn {
  completion: Ref<string>
  status: Ref<ChatStatus>
  error: Ref<Error | undefined>
  isLoading: Ref<boolean>
  id: string
  input: Ref<string>
  complete: (prompt?: string) => Promise<string>
  stop: () => void
  clearError: () => void
  handleSubmit: (event?: Event) => void
}

export function useCompletion(options: UseCompletionOptions): UseCompletionReturn {
  const id = options.id ?? generateId()
  const completion = ref('')
  const status = ref<ChatStatus>('ready') as Ref<ChatStatus>
  const error = ref<Error | undefined>(undefined)
  const input = ref('')

  let abortController: AbortController | null = null

  const complete = async (prompt?: string): Promise<string> => {
    const text = prompt ?? input.value
    if (!text.trim()) return completion.value

    if (!prompt) input.value = ''

    completion.value = ''
    error.value = undefined
    status.value = 'submitted'

    abortController?.abort()
    abortController = new AbortController()

    try {
      options.provider.validateConfig()

      const system = toValue(options.system)
      const messages = system
        ? [
          { role: 'system' as const, content: system },
          { role: 'user' as const, content: text },
        ]
        : [{ role: 'user' as const, content: text }]

      const { stream } = await tracedChat(options.provider, {
        model: toValue(options.model),
        messages,
        signal: abortController.signal,
      }, id)

      status.value = 'streaming'
      const reader = stream.getReader()
      let finishReason: FinishReason = 'stop'
      let usage: Usage | undefined

      while (true) {
        if (abortController.signal.aborted) break

        const { done, value } = await reader.read()
        if (done) break

        if (value.type === 'text-delta') {
          completion.value += value.text
        } else if (value.type === 'finish') {
          finishReason = value.finishReason
          usage = value.usage
        } else if (value.type === 'error') {
          throw value.error
        }
      }

      reader.releaseLock()
      status.value = 'ready'
      options.onFinish?.(completion.value, { finishReason, usage })

      return completion.value
    } catch (err) {
      if (!abortController || abortController.signal.aborted) {
        status.value = 'ready'
        return completion.value
      }
      const e = err instanceof Error ? err : new Error(String(err))
      error.value = e
      status.value = 'error'
      options.onError?.(e)
      return completion.value
    }
  }

  const stop = (): void => {
    abortController?.abort()
    abortController = null
    status.value = 'ready'
  }

  const clearError = (): void => {
    error.value = undefined
    if (status.value === 'error') status.value = 'ready'
  }

  const handleSubmit = (event?: Event): void => {
    event?.preventDefault?.()
    if (!input.value.trim()) return
    complete()
  }

  const isLoading = computed(() => status.value === 'submitted' || status.value === 'streaming')

  const unregister = registerDevtoolsSession({
    id,
    kind: 'completion',
    providerId: options.provider.id,
    getModel: () => toValue(options.model),
    status,
    error,
    output: completion,
    stop,
  })

  if (getCurrentScope()) {
    onScopeDispose(() => {
      abortController?.abort()
      unregister()
    })
  }

  return {
    completion,
    status: computed(() => status.value),
    error,
    isLoading,
    id,
    input,
    complete,
    stop,
    clearError,
    handleSubmit,
  }
}
