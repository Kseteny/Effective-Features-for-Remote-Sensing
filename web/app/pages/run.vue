<script setup lang="ts">
import type { Preset, Criterion, RunStatus, RunResult } from '~/types/api'

useHead({ title: 'Расчёт' })

const config = useRuntimeConfig()
const api = config.public.apiBase

const { data: presetsData } = await useFetch<{ items: Preset[] }>(`${api}/api/presets`)
const { data: criteriaData } = await useFetch<{ items: Criterion[] }>(`${api}/api/criteria`)

const preset = ref('thinned')
const chosen = ref<string[]>(['bhattacharyya', 'knn'])

const status = ref<RunStatus | null>(null)
const result = ref<RunResult | null>(null)
const errorText = ref('')
let timer: ReturnType<typeof setInterval> | null = null

const busy = computed(() =>
  status.value?.status === 'queued' || status.value?.status === 'running'
)

function label(id: string) {
  return id === 'knn' ? 'kNN' : 'Бхаттачарья'
}

function tagClass(id: string) {
  return id === 'knn' ? 'tag tag--knn' : 'tag tag--bhatta'
}

function stopTimer() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

async function start() {
  errorText.value = ''
  result.value = null
  status.value = null

  try {
    const started = await $fetch<{ task_id: string }>(`${api}/api/runs`, {
      method: 'POST',
      body: { preset: preset.value, criteria: chosen.value },
    })
    poll(started.task_id)
  } catch (e: any) {
    errorText.value = e?.data?.detail ?? 'Сервер расчётов не отвечает. Проверьте, запущен ли он.'
  }
}

function poll(taskId: string) {
  stopTimer()
  timer = setInterval(async () => {
    try {
      const s = await $fetch<RunStatus>(`${api}/api/runs/${taskId}`)
      status.value = s

      if (s.status === 'done') {
        stopTimer()
        result.value = await $fetch<RunResult>(`${api}/api/runs/${taskId}/result`)
      } else if (s.status === 'failed') {
        stopTimer()
        errorText.value = s.error ?? 'Расчёт прервался'
      }
    } catch {
      stopTimer()
      errorText.value = 'Связь с сервером потеряна'
    }
  }, 1500)
}

onUnmounted(stopTimer)
</script>

<template>
  <div class="page">
    <div class="page__head">
      <h1>Расчёт</h1>
      <p class="page__lead">
        Выберите объём выборки и критерии — программа отберёт признаки
        каждым из них и покажет, где результаты сошлись.
      </p>
    </div>

    <div class="run">
      <!-- Панель настроек -->
      <aside class="run__controls">
        <div class="card">
          <p class="card__title">Объём выборки</p>
          <label v-for="p in presetsData?.items" :key="p.id" class="choice">
            <input type="radio" :value="p.id" v-model="preset" :disabled="busy">
            <span>
              <span class="choice__name">{{ p.name }}</span>
              <span class="choice__note">{{ p.description }}</span>
            </span>
          </label>
        </div>

        <div class="card">
          <p class="card__title">Критерии отбора</p>
          <label v-for="c in criteriaData?.items" :key="c.id" class="choice">
            <input type="checkbox" :value="c.id" v-model="chosen" :disabled="busy">
            <span>
              <span class="choice__name">{{ c.name }}</span>
              <span class="choice__note">{{ c.description }}</span>
            </span>
          </label>
        </div>

        <button class="btn run__go" :disabled="busy || chosen.length === 0" @click="start">
          {{ busy ? 'Считаю…' : 'Запустить' }}
        </button>

        <p v-if="chosen.length === 0" class="mono-sm run__hint">
          Выберите хотя бы один критерий
        </p>
      </aside>

      <!-- Результаты -->
      <section class="run__output stack">
        <p v-if="errorText" class="notice notice--error">{{ errorText }}</p>

        <div v-if="status && busy" class="card">
          <p class="card__title">{{ status.stage }} · {{ status.elapsed_sec }} с</p>
          <div class="progress__bar">
            <div class="progress__fill" :style="{ width: (status.progress * 100) + '%' }"></div>
          </div>
          <pre class="log">{{ status.log_tail.join('\n') }}</pre>
        </div>

        <div v-if="!status && !result && !errorText" class="notice notice--empty">
          Результаты появятся здесь после запуска.
        </div>

        <template v-if="result">
          <div class="card">
            <p class="card__title">Итог</p>
            <p class="muted run__meta">
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
            <p v-if="result.agreement" class="muted run__agree">
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
        </template>
      </section>
    </div>
  </div>
</template>

<style scoped>
.run {
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  gap: 1.5rem;
  align-items: start;
}

.run__controls { position: sticky; top: 5.5rem; }
.run__controls .card + .card { margin-top: 1rem; }
.run__go { margin-top: 1rem; }
.run__hint { text-align: center; margin: 0.5rem 0 0; }
.run__meta { margin: 0 0 1rem; font-size: 0.9rem; }
.run__agree { margin: 1.25rem 0 0; font-size: 0.9rem; }

@media (max-width: 860px) {
  .run { grid-template-columns: 1fr; }
  .run__controls { position: static; }
}
</style>
