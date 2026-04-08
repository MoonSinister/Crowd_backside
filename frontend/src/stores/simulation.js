/**
 * stores/simulation.js — Pinia store for simulation state.
 */
import { defineStore } from 'pinia'
import {
  getSimulations,
  getSimulation,
  createSimulation,
  deleteSimulation,
  getTrajectories,
  getIntentStats,
} from '@/utils/api'

export const useSimulationStore = defineStore('simulation', {
  state: () => ({
    runs: [],
    currentRun: null,
    trajectories: [],
    intentStats: [],
    loading: false,
    error: null,
  }),

  actions: {
    async fetchRuns() {
      this.loading = true
      try {
        const { data } = await getSimulations()
        this.runs = data
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    },

    async fetchRun(id) {
      const { data } = await getSimulation(id)
      this.currentRun = data
      return data
    },

    async startSimulation(payload) {
      this.loading = true
      try {
        const { data } = await createSimulation(payload)
        this.runs.unshift(data)
        return data
      } catch (e) {
        this.error = e.message
        throw e
      } finally {
        this.loading = false
      }
    },

    async removeRun(id) {
      await deleteSimulation(id)
      this.runs = this.runs.filter((r) => r.id !== id)
    },

    async fetchTrajectories(runId, params = {}) {
      const { data } = await getTrajectories(runId, params)
      this.trajectories = data
    },

    async fetchIntentStats(runId) {
      const { data } = await getIntentStats(runId)
      this.intentStats = data
    },
  },
})
