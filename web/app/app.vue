<template>
  <div>
    <nav>
      <NuxtLink to="/">Признаки</NuxtLink>
      <NuxtLink to="/run">Запуск расчёта</NuxtLink>
    </nav>
    <NuxtPage />
  </div>
</template>

<style>
body { margin: 0; font-family: system-ui, sans-serif; }
nav {
  display: flex; gap: 1.5rem;
  padding: 1rem 2rem;
  border-bottom: 1px solid #ddd;
}
nav a { color: #333; text-decoration: none; }
nav a.router-link-active { font-weight: 600; color: #0066cc; }
</style>