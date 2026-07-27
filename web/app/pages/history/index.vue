<script setup lang="ts">
import type { HistoryResponse } from '~/types/api'

useHead({ title: 'История' })

const config = useRuntimeConfig()
const api = config.public.apiBase

const { data, refresh, error } = await useFetch<HistoryResponse>(`${api}/api/history`)

const presetNames: Record<string, string> = {
  fast: 'Быстрый',
  research: 'Исследовательский',
  thinned: 'Прореживание',
  full: 'Полный',
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('ru', {
    day: '2-digit', month: '2-digit',
    hour: '2-digit', minute: '2-digit',
  })
}

function label(id: string) {
  return id === 'knn' ? 'kNN' : 'Бхатт.'
}

function tagClass(id: string) {
  return id === 'knn' ? 'tag tag--knn' : 'tag tag--bhatta'
}

async function remove(taskId: string) {
  if (!confirm('Удалить этот расчёт из истории?')) return
  await $fetch(`${api}/api/history/${taskId}`, { method: 'DELETE' })
  await refresh()
}
</script>

<template>
  <div class="page">
    <div class="page__head">
      <h1>История</h1>
      <p class="page__lead">
        Все законченные расчёты. Сохраняются в базе, поэтому остаются
        на месте после перезапуска сервера.
      </p>
    </div>

    <p v-if="error" class="notice notice--error">
      Сервер расчётов не отвечает. Проверьте, запущен ли он на порту 8000.
    </p>

    <div v-else-if="!data || data.total === 0" class="notice notice--empty">
      Пока пусто. Первый расчёт можно запустить на странице
      <NuxtLink to="/run">«Расчёт»</NuxtLink>.
    </div>

    <div v-else class="card">
      <table class="table">
        <thead>
          <tr>
            <th>Когда</th>
            <th>Режим</th>
            <th>Пикселей</th>
            <th>Точность</th>
            <th>Время</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="it in data.items" :key="it.task_id">
            <td>
              <NuxtLink :to="`/history/${it.task_id}`" class="when">
                {{ formatDate(it.created_at) }}
              </NuxtLink>
            </td>
            <td>{{ presetNames[it.preset] ?? it.preset }}</td>
            <td class="num">{{ it.n_pixels?.toLocaleString('ru') ?? '—' }}</td>
            <td class="accs">
              <span v-for="(acc, crit) in it.accuracies" :key="crit" :class="tagClass(crit)">
                {{ label(crit) }} {{ (acc * 100).toFixed(1) }}%
              </span>
            </td>
            <td class="num">{{ it.total_time_sec }} с</td>
            <td>
              <button class="icon-btn" @click="remove(it.task_id)" :aria-label="'Удалить расчёт'">×</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.when { font-weight: 500; text-decoration: none; }
.when:hover { text-decoration: underline; }
.accs { display: flex; gap: 0.35rem; flex-wrap: wrap; }
</style>
