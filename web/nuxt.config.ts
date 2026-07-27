export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },

  css: ['~/assets/css/main.css'],

  app: {
    head: {
      htmlAttrs: { lang: 'ru' },
      titleTemplate: '%s — Отбор признаков',
      title: 'Отбор признаков',
      link: [
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        {
          rel: 'stylesheet',
          href: 'https://fonts.googleapis.com/css2?family=Onest:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap',
        },
      ],
    },
  },

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
