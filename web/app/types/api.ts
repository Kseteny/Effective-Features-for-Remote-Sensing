export interface Feature {
  index: number
  name: string
  group: 'spectral' | 'textural'
  window: number | null
  description: string
}

export interface FeaturesResponse {
  total: number
  spectral: number
  textural: number
  items: Feature[]
}

export interface Preset {
  id: string
  name: string
  description: string
}

export interface Criterion {
  id: string
  name: string
  type: 'filter' | 'wrapper'
  speed: string
  scope: 'pair' | 'all'
  unit: string
  description: string
  classifier_free: boolean
  color: string
}

export interface RunStatus {
  task_id: string
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
  stage: string
  progress: number
  log_tail: string[]
  elapsed_sec: number
  error: string | null
}

export interface QualityPoint {
  k: number
  accuracy: number
  f1_macro: number
}

export interface CriterionResult {
  id: string
  name?: string
  unit?: string
  color?: string
  /** Качество для каждого размера набора. У старых записей в истории нет. */
  quality_curve?: QualityPoint[]
  selected: number[]
  selected_names: string[]
  history: number[]
  accuracy: number
  error_rate: number
  f1_macro: number
  time_sec: number
}

export interface RunResult {
  task_id: string
  dataset: {
    n_pixels: number
    n_classes: number
    n_features: number
    feature_names: string[]
  }
  criteria: CriterionResult[]
  agreement: {
    both: string[]
    only_first: string[]
    only_second: string[]
  } | null
  total_time_sec: number
}

export interface HistoryItem {
  task_id: string
  created_at: string
  preset: string
  criteria: string[]
  n_pixels: number | null
  n_classes: number | null
  total_time_sec: number | null
  accuracies: Record<string, number>
}

export interface HistoryResponse {
  items: HistoryItem[]
  total: number
}

export interface DatasetClass {
  id: number
  name: string
}

export interface DatasetInfo {
  name: string
  n_bands: number
  band_order: string[]
  band_roles: Record<string, string>
  feature_bands: string[]
  n_spectral: number
  n_textural: number
  n_features: number
  n_classes: number
  classes: DatasetClass[]
  indices_available: string[]
  indices_missing: { name: string; needs: string[] }[]
  n_pairs: number | null
}

export interface DatasetReport {
  ok: boolean
  dataset?: DatasetInfo
  notes?: string[]
  error?: string
}
