<script setup lang="ts">
import type { RunResult } from '~/types/api'

const route = useRoute()
const config = useRuntimeConfig()
const api = config.public.apiBase

const taskId = route.params.id as string
useHead({ title: `Расчёт ${taskId}` })

const { data: result, error } = await useFetch<RunResult>(
  `${api}/api/history/${taskId}`
)

function label(id: string) {
  return id === 'knn' ? 'kNN' : 'Бхаттачарья'
}

function tagClass(id: string) {
  return id === 'knn' ? 'tag tag--knn' : 'tag tag--bhatta'
}
</script>

<template>
  <div class="page">
    <NuxtLink to="/history" class="back">← История</NuxtLink>

    <div class="page__head">
      <h1>Расчёт <span class="num id">{{ taskId }}</span></h1>
    </div>

    <p v-if="error" class="notice notice--error">
      Такого расчёта нет — возможно, его удалили из истории.
    </p>

    <div v-else-if="result" class="stack">
      <div class="card">
        <p class="card__title">Итог</p>
        <p class="muted meta">
          <span class="num">{{ result.dataset.n_pixels.toLocaleString('ru') }}</span> пикселей ·
          <span class="num">{{ result.dataset.n_classes }}</span> классов ·
          всего <span class="num">{{ result.total_time_sec }}</span> с
        </p>

        <table class="table">
          <thead>
            <tr>
              <th>Критерий</th>
              <th>Признаков</th>
              <th>Точность</th>
              <th>F1-macro</th>
              <th>Время</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="c in result.criteria" :key="c.id">
              <td><span :class="tagClass(c.id)">{{ label(c.id) }}</span></td>
              <td class="num">{{ c.selected.length }}</td>
              <td class="num">{{ (c.accuracy * 100).toFixed(1) }}%</td>
              <td class="num">{{ c.f1_macro.toFixed(3) }}</td>
              <td class="num">{{ c.time_sec }} с</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card">
        <p class="card__title">Что отобрано</p>
        <FeatureGrid
          :all-features="result.dataset.feature_names"
          :criteria="result.criteria"
        />
        <p v-if="result.agreement" class="muted agree">
          Совпало у обоих критериев:
          <span class="num">{{ result.agreement.both.length }}</span> из
          <span class="num">{{ result.dataset.n_features }}</span>
        </p>
      </div>

      <ClientOnly>
        <div v-for="c in result.criteria" :key="'ch-' + c.id" class="card">
          <p class="card__title">
            {{ c.id === 'knn'
              ? 'Точность по шагам отбора'
              : 'Расстояние Бхаттачарьи по шагам отбора' }}
          </p>
          <SelectionChart
            :labels="c.selected_names"
            :values="c.history"
            :y-label="c.id === 'knn' ? 'Точность, %' : 'D_B'"
            :as-percent="c.id === 'knn'"
            :color="c.id"
          />
        </div>
      </ClientOnly>
    </div>
  </div>
</template>

<style scoped>
.back {
  display: inline-block;
  margin-bottom: 1rem;
  font-size: 0.88rem;
  text-decoration: none;
  color: var(--ink-soft);
}

.back:hover { color: var(--forest); }
.id { font-size: 1.2rem; color: var(--ink-faint); font-weight: 400; }
.meta { margin: 0 0 1rem; font-size: 0.9rem; }
.agree { margin: 1.25rem 0 0; font-size: 0.9rem; }
</style>
