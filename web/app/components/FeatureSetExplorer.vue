<script setup lang="ts">
/**
 * Разбор набора признаков по его размеру.
 *
 * Отбор идёт по порядку: первый признак, второй, третий. Значит первые k
 * элементов списка - это и есть лучший набор из k признаков. Ползунок
 * пользуется этим свойством и позволяет посмотреть любой размер,
 * не пересчитывая ничего заново.
 *
 * Отвечает на вопрос, который ставил научный руководитель: «если признаков
 * три, то выше вот эти, если пять, то выше вот эти, и на каком качестве
 * это всё дело делается».
 */
interface QualityPoint {
  k: number
  accuracy: number
  f1_macro: number
}

interface CriterionResult {
  id: string
  name?: string
  unit?: string
  color?: string
  selected_names: string[]
  history?: number[]
  quality_curve?: QualityPoint[]
}

const props = defineProps<{
  allFeatures: string[]
  criteria: CriterionResult[]
}>()

const maxK = computed(() =>
  Math.max(1, ...props.criteria.map(c => c.selected_names.length))
)

const k = ref(maxK.value)

// Если пришёл другой результат - сдвигаем ползунок к новому максимуму
watch(maxK, v => { k.value = v })

/** Критерии с обрезанными до k наборами - для сетки признаков. */
const trimmed = computed(() =>
  props.criteria.map(c => ({
    ...c,
    selected_names: c.selected_names.slice(0, k.value),
  }))
)

/** Есть ли у нас данные о качестве. У старых записей в истории их нет. */
const hasQuality = computed(() =>
  props.criteria.some(c => (c.quality_curve?.length ?? 0) > 0)
)

interface Row {
  id: string
  name: string
  color?: string
  unit?: string
  count: number
  short: boolean
  value: number | null
  accuracy: number | null
  delta: number | null
}

const rows = computed<Row[]>(() =>
  props.criteria.map(c => {
    const count = Math.min(k.value, c.selected_names.length)
    const point = c.quality_curve?.[count - 1] ?? null
    const prev = count > 1 ? c.quality_curve?.[count - 2] ?? null : null
    return {
      id: c.id,
      name: c.name ?? c.id,
      color: c.color,
      unit: c.unit,
      count,
      short: c.selected_names.length < k.value,
      value: c.history?.[count - 1] ?? null,
      accuracy: point ? point.accuracy : null,
      delta: point && prev ? point.accuracy - prev.accuracy : null,
    }
  })
)

const CRITERION_HEX: Record<string, string> = {
  forest: '#14664A',
  gold: '#9C620F',
  plum: '#6B3A7A',
  water: '#276B96',
}

function tagStyle(color?: string) {
  if (!color) return {}
  return { background: `var(--${color}, ${CRITERION_HEX[color] ?? '#5A6B62'})` }
}

function formatValue(r: Row) {
  if (r.value === null) return '-'
  return r.unit === 'точность'
    ? `${(r.value * 100).toFixed(1)}%`
    : r.value.toFixed(3)
}
</script>

<template>
  <div>
    <div class="slider">
      <label for="k-range" class="slider__label">Размер набора</label>
      <input
        id="k-range"
        type="range"
        min="1"
        :max="maxK"
        step="1"
        v-model.number="k"
        class="slider__input"
      >
      <output class="slider__value num" for="k-range">{{ k }}</output>
    </div>

    <table class="table sizes">
      <thead>
        <tr>
          <th>Критерий</th>
          <th>Признаков</th>
          <th>Значение</th>
          <th v-if="hasQuality">Точность</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in rows" :key="r.id">
          <td><span class="tag tag--criterion" :style="tagStyle(r.color)">{{ r.name }}</span></td>
          <td class="num">
            {{ r.count }}
            <span v-if="r.short" class="muted note">(всего столько)</span>
          </td>
          <td class="num">{{ formatValue(r) }}</td>
          <td v-if="hasQuality" class="num">
            <template v-if="r.accuracy !== null">
              {{ (r.accuracy * 100).toFixed(1) }}%
              <span v-if="r.delta !== null" class="delta" :class="{ 'delta--down': r.delta < 0 }">
                {{ r.delta >= 0 ? '+' : '' }}{{ (r.delta * 100).toFixed(1) }}
              </span>
            </template>
            <template v-else>-</template>
          </td>
        </tr>
      </tbody>
    </table>

    <FeatureGrid :all-features="allFeatures" :criteria="trimmed" />
  </div>
</template>

<style scoped>
.slider {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.2rem;
}

.slider__label { font-size: 0.9rem; color: var(--ink-soft); white-space: nowrap; }

.slider__input {
  flex: 1;
  accent-color: var(--forest, #14664A);
  min-width: 120px;
}

.slider__value {
  font-family: var(--font-display);
  font-size: 1.5rem;
  font-weight: 700;
  min-width: 2ch;
  text-align: right;
  color: var(--forest, #14664A);
}

.sizes { margin-bottom: 1.6rem; }

.note { font-size: 0.75rem; margin-left: 0.3rem; }

/* Прирост точности от добавления последнего признака: показывает,
   насколько ещё имеет смысл увеличивать набор. */
.delta {
  font-size: 0.75rem;
  color: var(--forest, #14664A);
  margin-left: 0.35rem;
}

.delta--down { color: var(--danger, #A03528); }

@media (max-width: 560px) {
  .slider { flex-wrap: wrap; }
}
</style>
