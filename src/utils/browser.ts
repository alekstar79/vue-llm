export function isBrowserContext(): boolean {
    // Workers expose importScripts but do not have window, so they are browser contexts too.
    return typeof window !== 'undefined'
        || typeof (globalThis as typeof globalThis & { importScripts?: unknown }).importScripts === 'function'
}