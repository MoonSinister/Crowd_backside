<template>
  <div class="sim-view">
    <h1>Simulation View</h1>

    <div class="controls">
      <label>
        Run ID
        <input v-model.number="runId" type="number" min="1" placeholder="Run ID" />
      </label>
      <label>
        Day
        <input v-model.number="day" type="number" min="0" placeholder="Day" />
      </label>
      <button @click="load">Load</button>
    </div>

    <CityViewer
      v-if="store.trajectories.length"
      :trajectories="store.trajectories"
    />
    <p v-else class="hint">Enter a Run ID and click Load to display agent movements.</p>

    <section v-if="store.trajectories.length" class="card meta">
      <p>{{ store.trajectories.length }} records  |
         {{ uniqueAgents }} agents  |
         Fast-path intents: {{ fastCount }}  |
         Slow-path intents: {{ slowCount }}
      </p>
    </section>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useSimulationStore } from '@/stores/simulation'
import CityViewer from '@/components/CityViewer.vue'

const store = useSimulationStore()
const runId = ref(null)
const day = ref(null)

async function load() {
  if (!runId.value) return
  const params = {}
  if (day.value !== null && day.value >= 0) params.day = day.value
  await store.fetchTrajectories(runId.value, params)
}

const uniqueAgents = computed(() =>
  new Set(store.trajectories.map((t) => t.agent_id)).size
)
const fastCount = computed(() =>
  store.trajectories.filter((t) => t.intent_path === 'fast').length
)
const slowCount = computed(() =>
  store.trajectories.filter((t) => t.intent_path === 'slow').length
)
</script>

<style scoped>
.sim-view { padding: 1.5rem; }
h1 { margin-bottom: 1rem; }
.controls { display: flex; gap: 0.75rem; align-items: flex-end; margin-bottom: 1rem; }
label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
input { background: #0a0a1a; border: 1px solid #3a3a6a; border-radius: 4px;
        color: #e0e0e0; padding: 0.4rem 0.6rem; width: 120px; }
button { background: #3a5cff; color: #fff; border: none; border-radius: 4px;
         padding: 0.5rem 1.2rem; cursor: pointer; }
.hint { color: #666; margin-top: 2rem; text-align: center; }
.card.meta { background: #12122a; border: 1px solid #2a2a4a; border-radius: 6px;
             padding: 0.75rem 1rem; margin-top: 1rem; font-size: 0.9rem; }
</style>
