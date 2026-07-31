import type { RunStatus, RunResult } from '~/types/api'

/**
 * Состояние расчёта, живущее независимо от страницы.
 *
 * Раньше оно хранилось внутри компонента: стоило уйти на другую вкладку -
 * и страница забывала про расчёт. Сам расчёт при этом продолжался
 * на сервере, просто показать его было уже нечем.
 *
 * Теперь состояние в useState (переживает переходы между страницами),
 * а таймер опроса живёт в модуле и не останавливается при уходе.
 * Если же страницу перезагрузили или открыли в другой вкладке,
 * сервер сам расскажет, какие расчёты идут.
 */

// Таймер вне компонента: иначе при уходе со страницы он бы умирал вместе
// с ней. Модульная переменная безопасна, потому что страница расчёта
// работает только в браузере (ssr: false).
let timer: ReturnType<typeof setInterval> | null = null

export function useRun() {
  const config = useRuntimeConfig()
  const api = config.public.apiBase

  const taskId = useState<string | null>('run:taskId', () => null)
  const status = useState<RunStatus | null>('run:status', () => null)
  const result = useState<RunResult | null>('run:result', () => null)
  const errorText = useState<string>('run:error', () => '')
  const cancelling = useState<boolean>('run:cancelling', () => false)

  const busy = computed(() =>
    status.value?.status === 'queued' || status.value?.status === 'running'
  )
  const cancelled = computed(() => status.value?.status === 'cancelled')

  function stopTimer() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  function poll(id: string) {
    stopTimer()
    taskId.value = id
    timer = setInterval(async () => {
      try {
        const s = await $fetch<RunStatus>(`${api}/api/runs/${id}`)
        status.value = s

        if (s.status === 'done') {
          stopTimer()
          result.value = await $fetch<RunResult>(`${api}/api/runs/${id}/result`)
        } else if (s.status === 'cancelled') {
          stopTimer()
          cancelling.value = false
        } else if (s.status === 'failed') {
          stopTimer()
          cancelling.value = false
          errorText.value = s.error ?? 'Расчёт прервался'
        }
      } catch {
        stopTimer()
        errorText.value = 'Связь с сервером потеряна'
      }
    }, 1500)
  }

  async function start(preset: string, criteria: string[]) {
    errorText.value = ''
    result.value = null
    status.value = null
    cancelling.value = false

    try {
      const started = await $fetch<{ task_id: string }>(`${api}/api/runs`, {
        method: 'POST',
        body: { preset, criteria },
      })
      poll(started.task_id)
    } catch (e: any) {
      errorText.value =
        e?.data?.detail ?? 'Сервер расчётов не отвечает. Проверьте, запущен ли он.'
    }
  }

  async function cancel() {
    if (!taskId.value) return
    cancelling.value = true
    try {
      await $fetch(`${api}/api/runs/${taskId.value}/cancel`, { method: 'POST' })
    } catch {
      // Расчёт мог успеть закончиться сам - тогда отменять уже нечего
      cancelling.value = false
    }
  }

  /**
   * Вызывается при открытии страницы. Если расчёт уже идёт - подхватывает
   * его: либо тот, что помним, либо спрашивает сервер (после перезагрузки
   * страницы своя память пуста, а расчёт на сервере продолжается).
   */
  async function resume() {
    if (timer) return                       // уже следим

    if (taskId.value && busy.value) {
      poll(taskId.value)
      return
    }

    if (result.value || status.value) return   // уже есть чем показать

    try {
      const active = await $fetch<{ items: { task_id: string }[] }>(
        `${api}/api/runs`
      )
      const first = active.items[0]
      if (first) poll(first.task_id)
    } catch {
      // Сервер недоступен - молча, ошибку покажем при попытке запуска
    }
  }

  return {
    taskId, status, result, errorText, cancelling,
    busy, cancelled,
    start, cancel, resume,
  }
}
