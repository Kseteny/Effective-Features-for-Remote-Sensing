export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },

  runtimeConfig: {
    public: {
      apiBase: 'http://localhost:8000'
    }
  },

  routeRules: {
    // Документация собирается заранее, при сборке проекта (SSG).
    // На выходе — готовые .html файлы, их можно раздавать
    // с любого статического хостинга, сервер не нужен.
    '/docs/**': { prerender: true },

    // Инструмент работает только в браузере (SPA).
    // Собрать его заранее нельзя: содержимое зависит от выбора
    // пользователя и ответов API.
    '/run': { ssr: false },
  },
})