<script setup lang="ts">
import type { Preset, Criterion, RunStatus, RunResult } from '~/types/api'

const config = useRuntimeConfig()
const api = config.public.apiBase

// Справочники — грузим один раз при открытии страницы
const { data: presetsData } = await useFetch<{ items: Preset[] }>(`${api}/api/presets`)
const { data: criteriaData } = await useFetch<{ items: Criterion[] }>(`${api}/api/criteria`)

// Что выбрал пользователь
const preset = ref('thinned')
const chosen = ref<string[]>(['bhattacharyya', 'knn'])

// Состояние расчёта
const status = ref<RunStatus | null>(null)
const result = ref<RunResult | null>(null)
const errorText = ref('')
let timer: ReturnType<typeof setInterval> | null = null

const busy = computed(() =>
  status.value?.status === 'queued' || status.value?.status === 'running'
)

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
    errorText.value = e?.data?.detail ?? 'Не удалось запустить расчёт'
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
        errorText.value = s.error ?? 'Расчёт завершился с ошибкой'
      }
    } catch {
      stopTimer()
      errorText.value = 'Потеряна связь с сервером'
    }
  }, 1500)
}

// Если пользователь уйдёт со страницы — опрос надо прекратить,
// иначе таймер продолжит стучаться на сервер в пустоту.
onUnmounted(stopTimer)
</script>

<template>
  <div class="wrap">
    <h1>Запуск расчёта</h1>

    <section>
      <h2>Объём данных</h2>
      <label v-for="p in presetsData?.items" :key="p.id" class="row">
        <input type="radio" :value="p.id" v-model="preset" :disabled="busy">
        <span><b>{{ p.name }}</b> — {{ p.description }}</span>
      </label>
    </section>

    <section>
      <h2>Критерии отбора</h2>
      <label v-for="c in criteriaData?.items" :key="c.id" class="row">
        <input type="checkbox" :value="c.id" v-model="chosen" :disabled="busy">
        <span><b>{{ c.name }}</b> — {{ c.description }}</span>
      </label>
    </section>

    <button :disabled="busy || chosen.length === 0" @click="start">
      {{ busy ? 'Считаю…' : 'Запустить' }}
    </button>

    <p v-if="errorText" class="err">{{ errorText }}</p>

    <section v-if="status && busy" class="progress">
      <p><b>{{ status.stage }}</b> — {{ status.elapsed_sec }} с</p>
      <div class="bar"><div :style="{ width: (status.progress * 100) + '%' }"></div></div>
      <pre>{{ status.log_tail.join('\n') }}</pre>
    </section>

    <section v-if="result">
      <h2>Результат</h2>
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
    </section>
  </div>
</template>

<style scoped>
.wrap { padding: 2rem; max-width: 900px; }
section { margin: 1.5rem 0; }
h2 { font-size: 1.1rem; }
.row { display: flex; gap: .5rem; align-items: baseline; margin: .4rem 0; cursor: pointer; }
button {
  padding: .6rem 1.4rem; font-size: 1rem; cursor: pointer;
  background: #0066cc; color: #fff; border: 0; border-radius: 6px;
}
button:disabled { background: #999; cursor: default; }
.err { color: crimson; }
.progress .bar { height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
.progress .bar div { height: 100%; background: #0066cc; transition: width .3s; }
pre {
  background: #f5f5f5; padding: .8rem; border-radius: 6px;
  font-size: .8rem; overflow-x: auto; max-height: 200px;
}
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: .5rem; text-align: left; }
.picked { margin: .5rem 0; font-size: .9rem; }
</style>