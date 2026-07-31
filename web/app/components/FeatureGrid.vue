<script setup lang="ts">
/**
 * Сетка признаков.
 *
 * Показывает все признаки в той структуре, в которой они и считаются:
 * сначала спектральные, затем блоки по размерам окна. Отобранные
 * подсвечены цветом своего критерия.
 *
 * Признак, выбранный несколькими критериями сразу, делится на полосы -
 * по одной на каждый критерий. Раньше это работало только для двух
 * (половинки), теперь для любого количества: цвет делится на равные
 * доли с резкой границей, без плавного перехода.
 */
interface CriterionResult {
  id: string
  name?: string
  color?: string
  selected_names: string[]
}

const props = defineProps<{
  allFeatures: string[]
  criteria: CriterionResult[]
}>()

const FALLBACK_COLORS = ['forest', 'gold', 'plum', 'water']

// Запасные значения цветов.
// Без них любая опечатка в названии токена или незамеченная правка стилей
// делает ячейку белым текстом на прозрачном фоне - то есть невидимой,
// причём молча, без ошибок в консоли. С запасным значением цвет
// отработает в любом случае.
const HEX: Record<string, string> = {
  forest: '#14664A',
  gold: '#9C620F',
  plum: '#6B3A7A',
  water: '#276B96',
}

function paint(tone: string): string {
  return `var(--${tone}, ${HEX[tone] ?? '#5A6B62'})`
}

const GROUPS = [
  { key: 'spectral', label: 'Спектральные' },
  { key: '3', label: 'Окно 3×3' },
  { key: '5', label: 'Окно 5×5' },
  { key: '7', label: 'Окно 7×7' },
  { key: '9', label: 'Окно 9×9' },
]

function groupOf(name: string): string {
  if (name.startsWith('Norm_') || ['NDVI', 'NDWI', 'NDBI'].includes(name)) {
    return 'spectral'
  }
  const parts = name.split('_')
  return parts[parts.length - 1] ?? 'spectral'
}

/** Цвет критерия: из ответа сервера, иначе по порядку из запаса. */
function colorOf(c: CriterionResult, i: number): string {
  return c.color ?? FALLBACK_COLORS[i % FALLBACK_COLORS.length]!
}

const withColors = computed(() =>
  props.criteria.map((c, i) => ({ ...c, tone: colorOf(c, i) }))
)

/** Для каждого признака - какие критерии его выбрали. */
const picks = computed(() => {
  const map = new Map<string, { id: string; name: string; tone: string }[]>()
  for (const name of props.allFeatures) map.set(name, [])
  for (const c of withColors.value) {
    for (const name of c.selected_names) {
      map.get(name)?.push({ id: c.id, name: c.name ?? c.id, tone: c.tone })
    }
  }
  return map
})

const grouped = computed(() =>
  GROUPS.map(g => ({
    ...g,
    features: props.allFeatures.filter(f => groupOf(f) === g.key),
  })).filter(g => g.features.length > 0)
)

/** Заливка ячейки: сплошная для одного критерия, полосы для нескольких. */
function cellStyle(name: string) {
  const by = picks.value.get(name) ?? []
  if (by.length === 0) return {}
  if (by.length === 1) {
    return { background: paint(by[0]!.tone), color: '#fff', fontWeight: '500' }
  }
  const step = 100 / by.length
  const stops = by
    .map((c, i) => `${paint(c.tone)} ${i * step}% ${(i + 1) * step}%`)
    .join(', ')
  return {
    background: `linear-gradient(105deg, ${stops})`,
    color: '#fff',
    fontWeight: '500',
  }
}

function cellTitle(name: string) {
  const by = picks.value.get(name) ?? []
  if (by.length === 0) return `${name} - не отобран`
  return `${name} - ${by.map(c => c.name).join(', ')}`
}

/** Сколько признаков выбрали все критерии сразу. */
const commonCount = computed(() => {
  if (withColors.value.length < 2) return null
  let n = 0
  for (const by of picks.value.values()) {
    if (by.length === withColors.value.length) n++
  }
  return n
})
</script>

<template>
  <div>
    <div class="grid">
      <div v-for="g in grouped" :key="g.key" class="grid__group">
        <div class="grid__label">{{ g.label }}</div>
        <div class="grid__cells">
          <div
            v-for="f in g.features"
            :key="f"
            class="cell"
            :style="cellStyle(f)"
            :title="cellTitle(f)"
          >
            {{ f }}
          </div>
        </div>
      </div>
    </div>

    <div class="legend">
      <span v-for="c in withColors" :key="c.id" class="legend__item">
        <i class="swatch" :style="{ background: paint(c.tone) }"></i>{{ c.name ?? c.id }}
      </span>
      <span class="legend__item">
        <i class="swatch"></i>не отобран
      </span>
      <span v-if="withColors.length > 1" class="legend__note">
        Полосатая ячейка - признак выбрали несколько критериев
      </span>
    </div>

    <p v-if="commonCount !== null" class="summary">
      Выбрали все критерии сразу:
      <span class="num">{{ commonCount }}</span> из
      <span class="num">{{ allFeatures.length }}</span>
    </p>
  </div>
</template>

<style scoped>
.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 1.4rem 1.8rem;
}

.grid__group { min-width: 130px; }

.grid__label {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--ink-faint);
  margin-bottom: 0.5rem;
}

.grid__cells {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.cell {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  padding: 0.3rem 0.55rem;
  border-radius: var(--r-sm);
  background: var(--surface-sunk);
  color: var(--ink-faint);
  cursor: default;
  white-space: nowrap;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 1.1rem;
  margin-top: 1.3rem;
  font-size: 0.78rem;
  color: var(--ink-soft);
}

.legend__item { display: inline-flex; align-items: center; gap: 0.4rem; }

.legend__note {
  color: var(--ink-faint);
  font-size: 0.74rem;
}

.swatch {
  width: 11px;
  height: 11px;
  border-radius: 2px;
  background: var(--surface-sunk);
  display: inline-block;
  flex-shrink: 0;
}

.summary {
  margin: 1.1rem 0 0;
  font-size: 0.9rem;
  color: var(--ink-soft);
}
</style>
