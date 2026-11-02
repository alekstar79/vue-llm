# Contributing to vue-llm

Thanks for helping. Bug reports, docs fixes, translations and features are all welcome.

## Setup

```bash
git clone https://github.com/dadashi44/vue-llm.git
cd vue-llm
npm install
```

Node 20 or 22. If `npm install` fails on Node 20 with `Cannot read properties of null (reading 'edgesOut')`, use Node 22: that is an npm bug, not ours.

## Scripts

| Command | What it does |
|---------|--------------|
| `npm test` | Unit tests (Vue 3) |
| `npm run test:vue2` | Unit tests against Vue 2.7 |
| `npm run typecheck` | `vue-tsc`, including the Chat UI registry components |
| `npm run lint` | ESLint, including code style |
| `npm run lint:fix` | Fix what ESLint can fix, formatting included |
| `npm run build` | Build the package into `dist/` |
| `npm run docs:dev` | Docs site with live demos at http://localhost:5173 |
| `npm run dev` | Local playground app |

CI runs all of them, so run them before opening a pull request.

Two bots help on pull requests: [autofix.ci](https://autofix.ci) commits `npm run lint:fix` to your branch, so formatting fixes itself, and [pkg.pr.new](https://pkg.pr.new) comments an install command for a build of your branch.

## Project layout

- `src/composables`: `useChat`, `useCompletion`, `useObject`, `usePromptInput`, `useAutoScroll`
- `src/providers`: OpenAI, Anthropic, Gemini, Ollama, SSE and WebLLM providers
- `src/server.ts`: `createLLMHandler` and server tools
- `src/components`: `MessageParts`, `StreamMarkdown`
- `src/devtools.ts`: the Vue DevTools plugin
- `registry/chat`: shadcn-vue components, served as `r/chat.json` by the docs build
- `docs`: the VitePress site; `docs/{ru,zh,pt,es,ja}` are translations

## Pull requests

- Keep one change per pull request, with tests for behaviour changes.
- Changes to the public API need docs in `docs/guide` (English). Mention in the PR if the translations still need updating; maintainers can help.
- Dependency changes: run `npm ci` in a clean checkout before pushing, so the lockfile stays valid for CI.
- Use clear commit messages, e.g. `fix: …`, `feat: …`, `docs: …`.

## Translations

The English docs are the source. Translated pages keep the English heading anchors with `{#anchor}`, so links work in every language. Code blocks stay identical to the English ones.

## Reporting bugs and ideas

Use the [issue templates](https://github.com/dadashi44/vue-llm/issues/new/choose). For security issues see [SECURITY.md](./SECURITY.md).
