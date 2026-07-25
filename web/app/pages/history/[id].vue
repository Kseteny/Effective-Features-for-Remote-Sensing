<script setup lang="ts">
import type { RunResult } from '~/types/api'

const route = useRoute()
const config = useRuntimeConfig()
const api = config.public.apiBase

const taskId = route.params.id as string

const { data: result, error } = await useFetch<RunResult>(
  `${api}/api/history/${taskId}`
)
</script>

<template>
  <div class="wrap">
    <NuxtLink to="/history" class="back">← К списку</NuxtLink>

    <h1>Расчёт {{ taskId }}</h1>

    <p v-if="error" class="err">Запись не найдена.</p>

    <template v-else-if="result">
      <p>
        Выборка: {{ result.dataset.n_pixels.toLocaleString('ru') }} пикселей,
        {{ result.dataset.n_classes }} классов.
        Всего времени: {{ result.total_time_sec }} с
      </p>

      <table>
        <thead>
          <tr><th>Критерий</th><th>Признаков</th><th>Точность</th><th>F1-macro</th><th>Время</th></tr>
        </thead>
        <tbody>
          <tr v-for="c in result.criteria" :key="c.id">
            <td>{{ c.id }}</td>
            <td>{{ c.selected.length }}</td>
            <td>{{ (c.accuracy * 100).toFixed(1) }}%</td>
            <td>{{ c.f1_macro.toFixed(3) }}</td>
            <td>{{ c.time_sec }} с</td>
          </tr>
        </tbody>
      </table>

      <div v-for="c in result.criteria" :key="'l-' + c.id" class="picked">
        <b>{{ c.id }}:</b> {{ c.selected_names.join(', ') }}
      </div>

      <ClientOnly>
        <SelectionChart
          v-for="c in result.criteria"
          :key="'ch-' + c.id"
          :labels="c.selected_names"
          :values="c.history"
          :title="c.id === 'knn'
            ? 'Точность классификации по шагам отбора (kNN)'
            : 'Накопленное расстояние Бхаттачарьи по шагам отбора'"
          :y-label="c.id === 'knn' ? 'Точность, %' : 'D_B'"
          :as-percent="c.id === 'knn'"
        />
      </ClientOnly>

      <p v-if="result.agreement">
        <b>Совпало у обоих ({{ result.agreement.both.length }}):</b>
        {{ result.agreement.both.join(', ') }}
      </p>
    </template>
  </div>
</template>

<style scoped>
.wrap { padding: 2rem; max-width: 900px; }
.back { display: inline-block; margin-bottom: 1rem; color: #0066cc; text-decoration: none; }
.err { color: crimson; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: .5rem; text-align: left; }
.picked { margin: .5rem 0; font-size: .9rem; }
</style>