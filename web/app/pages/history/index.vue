<script setup lang="ts">
import type { HistoryResponse } from '~/types/api'

useHead({ title: 'История' })

const config = useRuntimeConfig()
const api = config.public.apiBase

const { data, refresh, error } = await useFetch<HistoryResponse>(`${api}/api/history`)

// Описания критериев нужны, чтобы показать в списке их названия и цвета:
// в самой истории хранятся только идентификаторы.
const { data: criteriaData } = await useFetch<{
  items: { id: string; name: string; color: string }[]
}>(`${api}/api/criteria`)

const criteriaById = computed(() => {
  const map = new Map<string, { name: string; color: string }>()
  for (const c of criteriaData.value?.items ?? []) {
    map.set(c.id, { name: c.name, color: c.color })
  }
  return map
})

/** Короткое имя для тесной таблицы: первое слово названия. */
function shortLabel(id: string) {
  const name = criteriaById.value.get(id)?.name
  if (!name) return id
  if (id === 'knn') return 'kNN'
  return name.split(' ').pop() ?? name
}

const CRITERION_HEX: Record<string, string> = {
  forest: '#14664A',
  gold: '#9C620F',
  plum: '#6B3A7A',
  water: '#276B96',
}

function tagStyle(id: string) {
  const color = criteriaById.value.get(id)?.color
  if (!color) return {}
  return { background: `var(--${color}, ${CRITERION_HEX[color] ?? '#5A6B62'})` }
}

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
            <td class="num">{{ it.n_pixels?.toLocaleString('ru') ?? '-' }}</td>
            <td class="accs">
              <span
                v-for="(acc, crit) in it.accuracies"
                :key="crit"
                class="tag tag--criterion"
                :style="tagStyle(crit)"
                :title="criteriaById.get(crit)?.name ?? crit"
              >
                {{ shortLabel(crit) }} {{ (acc * 100).toFixed(1) }}%
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
