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
          <div class="brand__name">Отбор признаков</div>
          <div class="brand__sub">{{ title }}</div>
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
