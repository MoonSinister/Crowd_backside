/**
 * utils/api.js — Axios client configured for the backend API.
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Simulations ───────────────────────────────────────────────────────────

export const getSimulations = () => api.get('/simulations/')
export const getSimulation = (id) => api.get(`/simulations/${id}`)
export const createSimulation = (payload) => api.post('/simulations/', payload)
export const deleteSimulation = (id) => api.delete(`/simulations/${id}`)
export const getTrajectories = (id, params = {}) =>
  api.get(`/simulations/${id}/trajectories`, { params })
export const getIntentStats = (id) => api.get(`/simulations/${id}/intents`)

export default api
