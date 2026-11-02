import { setupDevtoolsPlugin } from '@vue/devtools-api'
import { watch, type App, type Plugin } from 'vue'
import { setDevtoolsBridge, type DevtoolsEvent, type DevtoolsSession } from './core/devtools'
import type { UIMessage } from './types'

const INSPECTOR_ID = 'vue-llm'
const TIMELINE_ID = 'vue-llm:requests'
const GREEN = 0x42B883

export interface LLMDevtoolsOptions {
  /** Include the full message history in request timeline events. Default `true`; `false` logs only the count. */
  logMessages?: boolean
}

interface SessionStats {
  requests: number
  errors: number
  promptTokens: number
  completionTokens: number
  lastLatency?: number
  lastFirstChunk?: number
  lastTokensPerSecond?: number
}

interface RequestInfo {
  start: number
  firstChunk?: number
  sessionId: string
  model: string
}

/**
 * Vue DevTools integration: an "LLM" inspector with every useChat / useCompletion / useObject
 * and a timeline of requests, first-token latency, tool calls and token usage.
 *
 * ```ts
 * import { LLMDevtools } from 'vue-llm/devtools'
 * if (import.meta.env.DEV) app.use(LLMDevtools)
 * ```
 */
export const LLMDevtools: Plugin<[LLMDevtoolsOptions?]> = {
  install(app: App, options: LLMDevtoolsOptions = {}) {
    const sessions = new Map<string, DevtoolsSession>()
    const stats = new Map<string, SessionStats>()
    const requests = new Map<string, RequestInfo>()
    const stopWatchers = new Map<string, () => void>()

    // Timeline events are dropped until devtools connect; the inspector reads live state
    type Api = Parameters<Parameters<typeof setupDevtoolsPlugin>[1]>[0]
    let api: Api | undefined

    const statsOf = (id: string): SessionStats => {
      let s = stats.get(id)
      if (!s) stats.set(id, s = { requests: 0, errors: 0, promptTokens: 0, completionTokens: 0 })
      return s
    }

    let refreshQueued = false
    const refresh = (): void => {
      if (!api || refreshQueued) return
      refreshQueued = true
      // Coalesce bursts of streaming updates into one devtools refresh
      setTimeout(() => {
        refreshQueued = false
        api?.sendInspectorTree(INSPECTOR_ID)
        api?.sendInspectorState(INSPECTOR_ID)
      }, 100)
    }

    const timeline = (event: Parameters<Api['addTimelineEvent']>[0]['event']): void => {
      api?.addTimelineEvent({ layerId: TIMELINE_ID, event })
    }

    const handle = (event: DevtoolsEvent): void => {
      switch (event.type) {
        case 'session-added': {
          const { session } = event
          sessions.set(session.id, session)
          statsOf(session.id)
          const sources = [session.status, session.error, session.messages, session.output].filter(Boolean)
          stopWatchers.set(session.id, watch(sources as never[], refresh, { deep: true }))
          break
        }
        case 'session-removed':
          sessions.delete(event.sessionId)
          stats.delete(event.sessionId)
          stopWatchers.get(event.sessionId)?.()
          stopWatchers.delete(event.sessionId)
          break
        case 'request-start': {
          requests.set(event.requestId, { start: event.time, sessionId: event.sessionId, model: event.model })
          statsOf(event.sessionId).requests++
          timeline({
            time: event.time,
            title: 'request',
            subtitle: `${event.providerId} · ${event.model}`,
            groupId: event.requestId,
            data: {
              session: event.sessionId,
              provider: event.providerId,
              model: event.model,
              messages: options.logMessages === false ? event.messages.length : event.messages,
              tools: event.tools,
            },
          })
          break
        }
        case 'first-chunk': {
          const request = requests.get(event.requestId)
          if (request) request.firstChunk = event.time
          const latency = request ? event.time - request.start : undefined
          if (latency !== undefined) statsOf(event.sessionId).lastFirstChunk = latency
          timeline({
            time: event.time,
            title: 'first token',
            subtitle: latency !== undefined ? `${latency} ms` : undefined,
            groupId: event.requestId,
            data: { latency: latency !== undefined ? `${latency} ms` : undefined },
          })
          break
        }
        case 'tool-call':
          timeline({
            time: event.time,
            title: `tool: ${event.toolName}`,
            subtitle: event.args.length > 60 ? `${event.args.slice(0, 60)}…` : event.args,
            groupId: event.requestId,
            data: { toolCallId: event.toolCallId, name: event.toolName, args: parseArgs(event.args) },
          })
          break
        case 'tool-result':
          timeline({
            time: event.time,
            title: event.error ? 'tool error' : 'tool result',
            logType: event.error ? 'error' : 'default',
            groupId: event.requestId,
            data: { toolCallId: event.toolCallId, result: event.result, error: event.error },
          })
          break
        case 'request-end': {
          const request = requests.get(event.requestId)
          requests.delete(event.requestId)
          const duration = request ? event.time - request.start : undefined
          const s = statsOf(event.sessionId)
          s.lastLatency = duration
          if (event.usage) {
            s.promptTokens += event.usage.promptTokens
            s.completionTokens += event.usage.completionTokens
          }
          const streamSeconds = request?.firstChunk ? (event.time - request.firstChunk) / 1000 : 0
          const tokensPerSecond = event.usage && streamSeconds > 0
            ? Math.round(event.usage.completionTokens / streamSeconds)
            : undefined
          s.lastTokensPerSecond = tokensPerSecond
          timeline({
            time: event.time,
            title: 'finish',
            subtitle: [
              event.finishReason,
              duration !== undefined && `${duration} ms`,
              event.usage && `${event.usage.totalTokens} tokens`,
              tokensPerSecond && `${tokensPerSecond} tok/s`,
            ].filter(Boolean).join(' · '),
            groupId: event.requestId,
            data: {
              finishReason: event.finishReason,
              duration: duration !== undefined ? `${duration} ms` : undefined,
              usage: event.usage,
              tokensPerSecond,
              characters: event.textLength,
            },
          })
          break
        }
        case 'request-error': {
          requests.delete(event.requestId)
          if (!event.aborted) statsOf(event.sessionId).errors++
          timeline({
            time: event.time,
            title: event.aborted ? 'aborted' : 'error',
            subtitle: event.aborted ? undefined : event.error.message,
            logType: event.aborted ? 'warning' : 'error',
            groupId: event.requestId,
            data: { error: event.error.message },
          })
          break
        }
      }
      refresh()
    }

    setDevtoolsBridge({ emit: handle })

    setupDevtoolsPlugin({
      id: 'vue-llm',
      label: 'vue-llm',
      packageName: 'vue-llm',
      homepage: 'https://vue-llm.js.org/',
      logo: 'https://vue-llm.js.org/logo.svg',
      app,
    }, (devtoolsApi) => {
      api = devtoolsApi

      api.addInspector({
        id: INSPECTOR_ID,
        label: 'LLM',
        icon: 'smart_toy',
        treeFilterPlaceholder: 'Search sessions',
        noSelectionText: 'Select a useChat, useCompletion or useObject',
        nodeActions: [
          { icon: 'stop', tooltip: 'Stop generating', action: id => sessions.get(id)?.stop() },
          { icon: 'delete_sweep', tooltip: 'Clear messages', action: id => sessions.get(id)?.clear?.() },
        ],
      })

      api.addTimelineLayer({ id: TIMELINE_ID, label: 'LLM requests', color: GREEN })

      api.on.getInspectorTree((payload) => {
        if (payload.inspectorId !== INSPECTOR_ID) return
        const filter = payload.filter?.toLowerCase() ?? ''
        payload.rootNodes = [...sessions.values()]
          .filter(s => !filter || `${s.kind} ${s.id} ${s.getModel()}`.toLowerCase().includes(filter))
          .map(s => ({
            id: s.id,
            label: `${KIND_LABEL[s.kind]} #${s.id.length > 12 ? s.id.slice(0, 6) : s.id}`,
            tags: [
              { label: s.status.value, textColor: 0xFFFFFF, backgroundColor: STATUS_COLOR[s.status.value] },
              { label: s.getModel(), textColor: 0x000000, backgroundColor: 0xE5E7EB },
            ],
          }))
      })

      api.on.getInspectorState((payload) => {
        if (payload.inspectorId !== INSPECTOR_ID) return
        const session = sessions.get(payload.nodeId)
        if (!session) return
        const s = statsOf(session.id)

        payload.state = {
          state: [
            { key: 'status', value: session.status.value },
            { key: 'provider', value: session.providerId },
            { key: 'model', value: session.getModel() },
            { key: 'error', value: session.error.value?.message ?? null },
          ],
          usage: [
            { key: 'requests', value: s.requests },
            { key: 'errors', value: s.errors },
            { key: 'prompt tokens', value: s.promptTokens },
            { key: 'completion tokens', value: s.completionTokens },
            { key: 'last time to first token', value: s.lastFirstChunk !== undefined ? `${s.lastFirstChunk} ms` : null },
            { key: 'last duration', value: s.lastLatency !== undefined ? `${s.lastLatency} ms` : null },
            { key: 'last tokens/s', value: s.lastTokensPerSecond ?? null },
          ],
          ...(session.messages && {
            messages: session.messages.value.map((m, i) => ({ key: `${i} ${m.role}`, value: summarizeMessage(m) })),
          }),
          ...(session.output && { output: [{ key: 'value', value: session.output.value }] }),
        }
      })
    })
  },
}

const KIND_LABEL: Record<DevtoolsSession['kind'], string> = {
  chat: 'useChat',
  completion: 'useCompletion',
  object: 'useObject',
}

const STATUS_COLOR: Record<string, number> = {
  ready: 0x42B883,
  submitted: 0xF59E0B,
  streaming: 0x3B82F6,
  error: 0xEF4444,
}

function summarizeMessage(message: UIMessage): unknown {
  return message.parts.map((part) => {
    switch (part.type) {
      case 'text':
      case 'reasoning':
        return { [part.type]: part.text }
      case 'tool-call':
        return { tool: part.toolName, state: part.state, args: part.args, result: part.result, error: part.error }
      case 'file':
        return { file: part.filename ?? part.mediaType }
      case 'source':
        return { source: part.url }
    }
  })
}

function parseArgs(args: string): unknown {
  try {
    return JSON.parse(args)
  } catch {
    return args
  }
}
