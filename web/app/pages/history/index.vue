<script setup lang="ts">
import type { HistoryResponse } from '~/types/api'

const config = useRuntimeConfig()
const api = config.public.apiBase

const { data, refresh } = await useFetch<HistoryResponse>(`${api}/api/history`)

const presetNames: Record<string, string> = {
  fast: 'Быстрый',
  research: 'Исследовательский',
  thinned: 'Прореживание',
  full: 'Полный',
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('ru', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

async function remove(taskId: string) {
  if (!confirm('Удалить этот расчёт из истории?')) return
  await $fetch(`${api}/api/history/${taskId}`, { method: 'DELETE' })
  await refresh()
}
</script>

<template>
  <div class="wrap">
    <h1>История расчётов</h1>

    <p v-if="!data || data.total === 0">
      Пока пусто. Запустите расчёт на странице
      <NuxtLink to="/run">«Запуск расчёта»</NuxtLink>.
    </p>

    <table v-else>
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
            <NuxtLink :to="`/history/${it.task_id}`">
              {{ formatDate(it.created_at) }}
            </NuxtLink>
          </td>
          <td>{{ presetNames[it.preset] ?? it.preset }}</td>
          <td>{{ it.n_pixels?.toLocaleString('ru') ?? '—' }}</td>
          <td>
            <span v-for="(acc, crit) in it.accuracies" :key="crit" class="acc">
              {{ crit === 'knn' ? 'kNN' : 'Бхатт.' }} {{ (acc * 100).toFixed(1) }}%
            </span>
          </td>
          <td>{{ it.total_time_sec }} с</td>
          <td>
            <button class="del" @click="remove(it.task_id)" title="Удалить">×</button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.wrap { padding: 2rem; max-width: 900px; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th, td { border: 1px solid #ddd; padding: .5rem .7rem; text-align: left; }
th { background: #fafafa; font-weight: 600; }
.acc { display: inline-block; margin-right: .8rem; white-space: nowrap; }
.del {
  border: 0; background: none; cursor: pointer;
  font-size: 1.2rem; color: #999; line-height: 1;
}
.del:hover { color: crimson; }
</style>