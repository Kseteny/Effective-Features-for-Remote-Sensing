<script setup lang="ts">
import { Chart } from 'chart.js/auto'

const props = defineProps<{
  labels: string[]
  values: number[]
  title: string
  yLabel: string
  asPercent?: boolean
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

function draw() {
  if (!canvas.value) return
  chart?.destroy()

  const shown = props.asPercent
    ? props.values.map(v => v * 100)
    : props.values

  chart = new Chart(canvas.value, {
    type: 'line',
    data: {
      labels: props.labels,
      datasets: [{
        label: props.yLabel,
        data: shown,
        borderColor: '#0066cc',
        backgroundColor: 'rgba(0, 102, 204, .1)',
        pointRadius: 4,
        fill: true,
        tension: 0.2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: props.title },
        legend: { display: false },
        tooltip: {
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
        x: { title: { display: true, text: 'Добавленный признак' } },
        y: { title: { display: true, text: props.yLabel } },
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
.chart-box { height: 320px; margin: 1rem 0; }
</style>