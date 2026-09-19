import { getJson } from './api.ts'

export type ModelEntry = {
  name: string
  path: string
  quantizable: boolean
}

export type ModelsResponse = {
  models: ModelEntry[]
}

export const getModels = () => getJson<ModelsResponse>('/api/models')
