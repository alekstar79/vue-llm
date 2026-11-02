import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import pluginVue from 'eslint-plugin-vue'
import stylistic from '@stylistic/eslint-plugin'

export default tseslint.config(
  { ignores: ['dist/', 'node_modules/', 'playground/', 'docs/.vitepress/dist/', 'docs/.vitepress/cache/'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  stylistic.configs.customize({ indent: 2, quotes: 'single', semi: false, braceStyle: '1tbs', arrowParens: false }),
  {
    files: ['**/*.vue'],
    languageOptions: {
      parserOptions: { parser: tseslint.parser },
    },
  },
  {
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-explicit-any': 'warn',
      'vue/multi-word-component-names': 'off',
      'vue/no-v-html': 'off',
      'vue/max-attributes-per-line': 'off',
      // Match the existing code: `{ a: string; b: number }` inline, no delimiters in multiline types
      '@stylistic/member-delimiter-style': ['error', {
        multiline: { delimiter: 'none' },
        singleline: { delimiter: 'semi', requireLast: false },
      }],
      '@stylistic/indent': ['error', 2, { flatTernaryExpressions: true, offsetTernaryExpressions: false }],
      // Chained ternaries, one-liners and both arrow-parens styles are already mixed through the code
      '@stylistic/arrow-parens': 'off',
      '@stylistic/multiline-ternary': 'off',
      '@stylistic/max-statements-per-line': 'off',
      '@stylistic/operator-linebreak': 'off',
    },
  },
)
