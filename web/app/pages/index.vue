<script setup lang="ts">
import type { FeaturesResponse, Feature } from '~/types/api'

useHead({ title: 'Признаки' })

const config = useRuntimeConfig()
const { data, error } = await useFetch<FeaturesResponse>(
  `${config.public.apiBase}/api/features`
)

/** Раскладываем признаки по группам в том же порядке, в каком они считаются. */
const groups = computed(() => {
  const items = data.value?.items ?? []
  const spectral = items.filter(f => f.group === 'spectral')
  const windows = [...new Set(items.filter(f => f.window).map(f => f.window))]

  const result: { label: string; features: Feature[] }[] = []
  if (spectral.length) result.push({ label: 'Спектральные', features: spectral })
  for (const w of windows) {
    result.push({
      label: `Текстурные, окно ${w}×${w}`,
      features: items.filter(f => f.window === w),
    })
  }
  return result
})
</script>

<template>
  <div class="page">
    <div class="page__head">
      <h1>Признаки</h1>
      <p class="page__lead">
        Всё, что программа вычисляет по снимку. Номер слева - тот же,
        что используется в выводе программы.
      </p>
    </div>

    <p v-if="error" class="notice notice--error">
      Сервер расчётов не отвечает. Проверьте, запущен ли он на порту 8000.
    </p>

    <template v-else-if="data">
      <div class="card summary">
        <div class="summary__item">
          <div class="summary__num num">{{ data.total }}</div>
          <div class="summary__label">всего</div>
        </div>
        <div class="summary__item">
          <div class="summary__num num">{{ data.spectral }}</div>
          <div class="summary__label">спектральных</div>
        </div>
        <div class="summary__item">
          <div class="summary__num num">{{ data.textural }}</div>
          <div class="summary__label">текстурных</div>
        </div>
      </div>

      <div v-for="g in groups" :key="g.label" class="card">
        <p class="card__title">{{ g.label }}</p>
        <table class="table">
          <tbody>
            <tr v-for="f in g.features" :key="f.index">
              <td class="num idx">{{ f.index }}</td>
              <td class="num name">{{ f.name }}</td>
              <td class="muted">{{ f.description }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>

<style scoped>
.summary { display: flex; gap: 2.5rem; }

.summary__num {
  font-family: var(--font-display);
  font-size: 1.8rem;
  font-weight: 700;
  line-height: 1;
  color: var(--forest);
}

.summary__label {
  font-size: 0.78rem;
  color: var(--ink-faint);
  margin-top: 0.3rem;
}

.idx { color: var(--ink-faint); width: 3rem; }
.name { font-weight: 500; width: 9rem; }

@media (max-width: 600px) {
  .name { width: auto; }
  .summary { gap: 1.5rem; }
}
</style>
