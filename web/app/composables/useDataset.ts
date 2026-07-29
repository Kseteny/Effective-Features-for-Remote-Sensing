import type { DatasetReport } from '~/types/api'

/**
 * Описание подключённого датасета.
 *
 * Запрашивается один раз и хранится в общем состоянии: название нужно
 * и в шапке сайта, и на странице «Данные», а дёргать сервер дважды
 * незачем.
 *
 * Загрузка вызывается только в браузере (onMounted). Иначе сборка
 * статических страниц документации потребовала бы запущенного сервера
 * расчётов — а они с ним никак не связаны.
 */
export function useDataset() {
  const config = useRuntimeConfig()
  const api = config.public.apiBase

  const report = useState<DatasetReport | null>('dataset:report', () => null)
  const failed = useState<string | null>('dataset:failed', () => null)
  const loading = useState<boolean>('dataset:loading', () => false)

  /** Название для шапки: из описания, иначе нейтральная подпись. */
  const title = computed(() =>
    report.value?.ok
      ? report.value.dataset?.name ?? 'Датасет без названия'
      : 'Данные не подключены'
  )

  async function load(force = false) {
    if (!force && (report.value || loading.value)) return
    loading.value = true
    failed.value = null
    try {
      report.value = await $fetch<DatasetReport>(`${api}/api/dataset`)
    } catch {
      report.value = null
      failed.value = 'Сервер расчётов не отвечает'
    } finally {
      loading.value = false
    }
  }

  return { report, failed, loading, title, load }
}
