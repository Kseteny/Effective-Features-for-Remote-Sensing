<script setup lang="ts">
/**
 * Сетка признаков.
 *
 * Показывает все 41 признак в той структуре, в которой они и считаются:
 * сначала спектральные, затем четыре блока по размерам окна. Отобранные
 * подсвечены цветом своего критерия, а те, что выбрали оба, — половинками
 * обоих цветов. Так видно и состав каждого набора, и их пересечение —
 * то есть ровно то, ради чего критерии и сравниваются.
 */
const props = defineProps<{
  allFeatures: string[]
  criteria: { id: string; selected_names: string[] }[]
}>()

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

/** Для каждого признака — какие критерии его выбрали. */
const picks = computed(() => {
  const map = new Map<string, string[]>()
  for (const name of props.allFeatures) map.set(name, [])
  for (const c of props.criteria) {
    for (const name of c.selected_names) {
      const list = map.get(name)
      if (list) list.push(c.id)
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

function cellClass(name: string) {
  const by = picks.value.get(name) ?? []
  if (by.length === 0) return 'cell'
  if (by.length > 1) return 'cell cell--both'
  return by[0] === 'knn' ? 'cell cell--knn' : 'cell cell--bhatta'
}

function cellTitle(name: string) {
  const by = picks.value.get(name) ?? []
  if (by.length === 0) return `${name} — не отобран`
  const names = by.map(id => (id === 'knn' ? 'kNN' : 'Бхаттачарья'))
  return `${name} — ${names.join(' и ')}`
}

const hasTwo = computed(() => props.criteria.length > 1)
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
            :class="cellClass(f)"
            :title="cellTitle(f)"
          >
            {{ f }}
          </div>
        </div>
      </div>
    </div>

    <div class="legend">
      <span class="legend__item"><i class="swatch swatch--bhatta"></i>Бхаттачарья</span>
      <span v-if="hasTwo" class="legend__item"><i class="swatch swatch--knn"></i>kNN</span>
      <span v-if="hasTwo" class="legend__item"><i class="swatch swatch--both"></i>оба критерия</span>
      <span class="legend__item"><i class="swatch"></i>не отобран</span>
    </div>
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

/* Отобранные заливаются сплошным цветом своего критерия — так сетка
   читается с одного взгляда, а не требует вглядываться в оттенки. */
.cell--bhatta { background: var(--forest); color: #fff; font-weight: 500; }
.cell--knn    { background: var(--gold);   color: #fff; font-weight: 500; }

/* Выбранный обоими: половина одного цвета, половина другого,
   резкая граница — сразу видно, что критерии сошлись. */
.cell--both {
  background: linear-gradient(105deg, var(--forest) 50%, var(--gold) 50%);
  color: #fff;
  font-weight: 500;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.1rem;
  margin-top: 1.3rem;
  font-size: 0.78rem;
  color: var(--ink-soft);
}

.legend__item { display: inline-flex; align-items: center; gap: 0.4rem; }

.swatch {
  width: 11px;
  height: 11px;
  border-radius: 2px;
  background: var(--surface-sunk);
  display: inline-block;
}

.swatch--bhatta { background: var(--forest); }
.swatch--knn    { background: var(--gold); }
.swatch--both   { background: linear-gradient(105deg, var(--forest) 50%, var(--gold) 50%); }
</style>
