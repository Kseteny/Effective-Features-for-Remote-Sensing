<script setup lang="ts">
import { Chart } from 'chart.js/auto'

const props = defineProps<{
  labels: string[]
  values: number[]
  yLabel: string
  asPercent?: boolean
  /** Идентификатор критерия — от него зависит цвет линии,
   *  чтобы график совпадал по цвету с меткой в таблице. */
  color?: string
}>()

const PALETTE: Record<string, { line: string; fill: string }> = {
  knn:           { line: '#9C620F', fill: 'rgba(156, 98, 15, 0.14)' },
  bhattacharyya: { line: '#14664A', fill: 'rgba(20, 102, 74, 0.14)' },
}

const canvas = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

function draw() {
  if (!canvas.value) return
  chart?.destroy()

  const tone = PALETTE[props.color ?? 'bhattacharyya'] ?? PALETTE.bhattacharyya!
  const shown = props.asPercent ? props.values.map(v => v * 100) : props.values

  chart = new Chart(canvas.value, {
    type: 'line',
    data: {
      labels: props.labels,
      datasets: [{
        label: props.yLabel,
        data: shown,
        borderColor: tone.line,
        backgroundColor: tone.fill,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: '#fff',
        pointBorderColor: tone.line,
        pointBorderWidth: 2,
        fill: true,
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#17211C',
          padding: 10,
          titleFont: { family: 'IBM Plex Mono', size: 12 },
          bodyFont: { family: 'IBM Plex Mono', size: 12 },
          displayColors: false,
          callbacks: {
            title: (items) => {
              const first = items[0]
              if (!first) return ''
              return `Шаг ${first.dataIndex + 1}: ${first.label}`
            },
            label: (item) => {
              const y = item.parsed.y ?? 0
              return props.asPercent
                ? `${props.yLabel}: ${y.toFixed(1)}%`
                : `${props.yLabel}: ${y.toFixed(3)}`
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 }, color: '#8A968E' },
        },
        y: {
          border: { display: false },
          grid: { color: '#E4EBE3' },
          ticks: { font: { family: 'IBM Plex Mono', size: 10 }, color: '#8A968E' },
          title: { display: true, text: props.yLabel, color: '#8A968E',
                   font: { family: 'IBM Plex Sans', size: 11 } },
        },
      },
    },
  })
}

onMounted(draw)
watch(() => [props.labels, props.values], draw, { deep: true })

// Chart.js держит обработчики событий и таймеры анимации.
// Без destroy() при уходе со страницы они останутся висеть в памяти.
onUnmounted(() => chart?.destroy())
</script>

<template>
  <div class="chart-box">
    <canvas ref="canvas"></canvas>
  </div>
</template>

<style scoped>
.chart-box { height: 300px; }
</style>
