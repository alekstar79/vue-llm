import { addComponent, addImports, addPluginTemplate, createResolver, defineNuxtModule, tryResolveModule } from '@nuxt/kit'

export interface ModuleOptions {
  /** Add the vue-llm inspector and timeline to Vue DevTools in development. Default `true`. */
  devtools?: boolean
}

export default defineNuxtModule({
  meta: {
    name: 'vue-llm',
    configKey: 'vueLlm',
    compatibility: { nuxt: '>=3.5.0' },
  },
  defaults: { devtools: true },
  // Nuxt's types come from @nuxt/schema, which is not a dependency, so describe what we use
  async setup(
    options: ModuleOptions,
    nuxt: { options: { runtimeConfig: Record<string, unknown>; css: string[]; dev: boolean; modulesDir: string[] } },
  ) {
    const resolver = createResolver(import.meta.url)

    addImports([
      { name: 'useChat', from: resolver.resolve('./index') },
      { name: 'useCompletion', from: resolver.resolve('./index') },
      { name: 'useObject', from: resolver.resolve('./index') },
      { name: 'usePrompt', from: resolver.resolve('./index') },
      { name: 'createPrompt', from: resolver.resolve('./index') },
      { name: 'createChatPrompt', from: resolver.resolve('./index') },
      { name: 'defineTool', from: resolver.resolve('./index') },
      { name: 'usePromptInput', from: resolver.resolve('./index') },
      { name: 'useAutoScroll', from: resolver.resolve('./index') },
    ])

    addImports([
      { name: 'createOpenAI', from: resolver.resolve('./providers') },
      { name: 'createAnthropic', from: resolver.resolve('./providers') },
      { name: 'createGoogle', from: resolver.resolve('./providers') },
      { name: 'createOllama', from: resolver.resolve('./providers') },
      { name: 'createSSEProvider', from: resolver.resolve('./providers') },
      { name: 'createWebLLM', from: resolver.resolve('./providers') },
      { name: 'isWebGPUSupported', from: resolver.resolve('./providers') },
      { name: 'detectWebGPUPrecision', from: resolver.resolve('./providers') },
    ])

    // Lowest priority, so an app component with the same name wins
    for (const name of ['StreamMarkdown', 'MessageParts']) {
      addComponent({ name, export: name, filePath: resolver.resolve('./components'), priority: -1 })
    }
    nuxt.options.css.push(resolver.resolve('./vue-llm.css'))

    if (nuxt.options.dev && options.devtools && await tryResolveModule('@vue/devtools-api', nuxt.options.modulesDir)) {
      const devtools = resolver.resolve('./devtools')
      addPluginTemplate({
        filename: 'vue-llm-devtools.client.mjs',
        mode: 'client',
        getContents: () => [
          `import { LLMDevtools } from ${JSON.stringify(devtools)}`,
          'export default defineNuxtPlugin(nuxtApp => { nuxtApp.vueApp.use(LLMDevtools) })',
        ].join('\n'),
      })
    }

    nuxt.options.runtimeConfig.vueLlm = {
      openaiApiKey: '',
      anthropicApiKey: '',
      ...nuxt.options.runtimeConfig.vueLlm as Record<string, string> | undefined,
    }
  },
})
