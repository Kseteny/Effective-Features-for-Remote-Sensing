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