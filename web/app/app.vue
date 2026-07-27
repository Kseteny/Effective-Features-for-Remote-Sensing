<template>
  <div>
    <header class="site-head">
      <div class="site-head__inner">
        <NuxtLink to="/" class="brand">
          <div class="brand__name">Отбор признаков</div>
          <div class="brand__sub">Sentinel-2 · MultiSenGE</div>
        </NuxtLink>

        <nav class="site-nav">
          <NuxtLink to="/run">Расчёт</NuxtLink>
          <NuxtLink to="/history">История</NuxtLink>
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
