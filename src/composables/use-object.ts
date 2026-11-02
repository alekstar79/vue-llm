import { computed, getCurrentScope, onScopeDispose, ref, type Ref } from 'vue'
import { type MaybeRefOrGetter, toValue } from '../utils/vue'
import type { ChatStatus, FinishReason, Usage } from '../types'
import type { LLMProvider } from '../providers/types'
import { generateId } from '../utils/generate-id'
import { parsePartialJSON } from '../core/partial-json'
import { registerDevtoolsSession, tracedChat } from '../core/devtools'

export interface UseObjectOptions<T> {
  id?: string
  provider: LLMProvider
  model: MaybeRefOrGetter<string>
  system?: MaybeRefOrGetter<string | undefined>
  schema: Record<string, unknown>
  onFinish?: (object: T, context: { finishReason: FinishReason; usage?: Usage }) => void
  onError?: (error: Error) => void
}

export interface UseObjectReturn<T> {
  object: Ref<T | undefined>
  partialObject: Ref<Partial<T> | undefined>
  status: Ref<ChatStatus>
  error: Ref<Error | undefined>
  isLoading: Ref<boolean>
  id: string
  input: Ref<string>
  submit: (prompt?: string) => Promise<T | undefined>
  stop: () => void
  clearError: () => void
}

export function useObject<T = unknown>(options: UseObjectOptions<T>): UseObjectReturn<T> {
  const id = options.id ?? generateId()
  const object = ref<T | undefined>(undefined) as Ref<T | undefined>
  const partialObject = ref<Partial<T> | undefined>(undefined) as Ref<Partial<T> | undefined>
  const status = ref<ChatStatus>('ready') as Ref<ChatStatus>
  const error = ref<Error | undefined>(undefined)
  const input = ref('')

  let abortController: AbortController | null = null

  const submit = async (prompt?: string): Promise<T | undefined> => {
    const text = prompt ?? input.value
    if (!text.trim()) return object.value

    if (!prompt) input.value = ''

    object.value = undefined
    partialObject.value = undefined
    error.value = undefined
    status.value = 'submitted'

    abortController?.abort()
    // Keep this request's signal so a stopped request cannot read a cleared controller.
    const requestController = new AbortController()
    abortController = requestController

    try {
      options.provider.validateConfig()

      const systemPrompt = [
        toValue(options.system) ?? '',
        'Respond with valid JSON matching this schema:',
        JSON.stringify(options.schema, null, 2),
        'Output ONLY the JSON object, no other text.',
      ].filter(Boolean).join('\n\n')

      const { stream } = await tracedChat(options.provider, {
        model: toValue(options.model),
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: text },
        ],
        signal: requestController.signal,
        providerOptions: {
          response_format: { type: 'json_object' },
        },
      }, id)

      if (requestController.signal.aborted) return object.value

      status.value = 'streaming'
      const reader = stream.getReader()
      let accumulated = ''
      let finishReason: FinishReason = 'stop'
      let usage: Usage | undefined

      while (true) {
        if (requestController.signal.aborted) break

        const { done, value } = await reader.read()
        if (requestController.signal.aborted) break
        if (done) break

        if (value.type === 'text-delta') {
          accumulated += value.text
          const partial = parsePartialJSON(accumulated)
          if (partial !== null && typeof partial === 'object') partialObject.value = partial as Partial<T>
        } else if (value.type === 'finish') {
          finishReason = value.finishReason
          usage = value.usage
        } else if (value.type === 'error') {
          throw value.error
        }
      }

      reader.releaseLock()
      if (requestController.signal.aborted) return object.value

      try {
        object.value = JSON.parse(accumulated) as T
        partialObject.value = object.value as Partial<T>
      } catch {
        throw new Error(`Failed to parse JSON response: ${accumulated.slice(0, 200)}`)
      }

      status.value = 'ready'
      options.onFinish?.(object.value, { finishReason, usage })

      return object.value
    } catch (err) {
      if (requestController.signal.aborted) return object.value
      const e = err instanceof Error ? err : new Error(String(err))
      error.value = e
      status.value = 'error'
      options.onError?.(e)
      return undefined
    } finally {
      if (abortController === requestController) abortController = null
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

  const isLoading = computed(() => status.value === 'submitted' || status.value === 'streaming')

  const unregister = registerDevtoolsSession({
    id,
    kind: 'object',
    providerId: options.provider.id,
    getModel: () => toValue(options.model),
    status,
    error,
    output: partialObject,
    stop,
  })

  if (getCurrentScope()) {
    onScopeDispose(() => {
      abortController?.abort()
      unregister()
    })
  }

  return {
    object,
    partialObject,
    status: computed(() => status.value),
    error,
    isLoading,
    id,
    input,
    submit,
    stop,
    clearError,
  }
}
