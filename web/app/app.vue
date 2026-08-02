<script setup lang="ts">
const { title, load } = useDataset()

// Название датасета подтягиваем только в браузере: иначе сборка
// статических страниц документации потребовала бы запущенного
// сервера расчётов, а он к ним отношения не имеет.
onMounted(() => load())
</script>

<template>
  <div>
    <header class="site-head">
      <div class="site-head__inner">
        <NuxtLink to="/run" class="brand">
          <svg class="brand__mark" width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <rect x="0" y="0" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.85" />
            <rect x="6.5" y="0" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.5" />
            <rect x="13" y="0" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.3" />
            <rect x="0" y="6.5" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.5" />
            <rect x="6.5" y="6.5" width="5" height="5" rx="1" fill="var(--gold)" />
            <rect x="13" y="6.5" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.3" />
            <rect x="0" y="13" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.3" />
            <rect x="6.5" y="13" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.5" />
            <rect x="13" y="13" width="5" height="5" rx="1" fill="var(--forest)" opacity="0.85" />
          </svg>
          <div class="brand__text">
            <div class="brand__name">Отбор признаков</div>
            <div class="brand__sub">{{ title }}</div>
          </div>
        </NuxtLink>

        <nav class="site-nav">
          <NuxtLink to="/run">Расчёт</NuxtLink>
          <NuxtLink to="/history">История</NuxtLink>
          <NuxtLink to="/data">Данные</NuxtLink>
          <NuxtLink to="/">Признаки</NuxtLink>
          <span class="site-nav__gap"></span>
          <NuxtLink to="/docs">О проекте</NuxtLink>
          <NuxtLink to="/docs/methods">Методы</NuxtLink>
        </nav>
      </div>
    </header>

    <main>
      <NuxtPage />
    </main>
  </div>
</template>
