<template>
  <div class="dashboard">
    <h1>Simulation Dashboard</h1>

    <!-- New simulation form -->
    <section class="card">
      <h2>New Simulation</h2>
      <form @submit.prevent="startSim">
        <label>
          Name
          <input v-model="form.name" required placeholder="Experiment name" />
        </label>
        <label>
          Agents
          <input v-model.number="form.num_agents" type="number" min="1" max="1000" />
        </label>
        <label>
          Steps
          <input v-model.number="form.num_steps" type="number" min="1" max="1440" />
        </label>
        <button type="submit" :disabled="store.loading">
          {{ store.loading ? 'Starting...' : 'Start Simulation' }}
        </button>
      </form>
    </section>

    <!-- Run list -->
    <section class="card">
      <h2>Simulation Runs</h2>
      <p v-if="store.error" class="error">{{ store.error }}</p>
      <table v-if="store.runs.length">
        <thead>
          <tr>
            <th>ID</th><th>Name</th><th>Agents</th><th>Steps</th>
            <th>Status</th><th>Started</th><th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="run in store.runs" :key="run.id">
            <td>{{ run.id }}</td>
            <td>{{ run.name }}</td>
            <td>{{ run.num_agents }}</td>
            <td>{{ run.num_steps }}</td>
            <td>
              <span :class="['badge', run.status]">{{ run.status }}</span>
            </td>
            <td>{{ formatDate(run.started_at) }}</td>
            <td>
              <router-link :to="`/simulation?run=${run.id}`">View</router-link>
              &nbsp;
              <button class="btn-danger" @click="removeRun(run.id)">Delete</button>
            </td>
          </tr>
        </tbody>
      </table>
      <p v-else>No simulation runs yet.</p>
    </section>
  </div>
</template>

<script setup>
import { reactive, onMounted } from 'vue'
import { useSimulationStore } from '@/stores/simulation'

const store = useSimulationStore()
const form = reactive({ name: 'Run 1', num_agents: 50, num_steps: 144 })

onMounted(() => store.fetchRuns())

async function startSim() {
  await store.startSimulation({ ...form })
}

async function removeRun(id) {
  if (confirm('Delete this run?')) await store.removeRun(id)
}

function formatDate(dt) {
  return dt ? new Date(dt).toLocaleString() : '—'
}
</script>

<style scoped>
.dashboard { padding: 1.5rem; max-width: 1100px; margin: 0 auto; }
h1 { margin-bottom: 1.5rem; font-size: 1.6rem; }
.card { background: #12122a; border: 1px solid #2a2a4a; border-radius: 8px;
        padding: 1.25rem; margin-bottom: 1.5rem; }
h2 { margin-bottom: 1rem; font-size: 1.1rem; color: #aac4ff; }
form { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: flex-end; }
label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
input { background: #0a0a1a; border: 1px solid #3a3a6a; border-radius: 4px;
        color: #e0e0e0; padding: 0.4rem 0.6rem; width: 160px; }
button { background: #3a5cff; color: #fff; border: none; border-radius: 4px;
         padding: 0.5rem 1.2rem; cursor: pointer; }
button:disabled { opacity: 0.5; }
.btn-danger { background: #c0392b; font-size: 0.8rem; padding: 0.25rem 0.6rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #2a2a4a; }
th { color: #7b9fff; }
.badge { padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; }
.badge.done { background: #1a4a2a; color: #4caf50; }
.badge.running { background: #1a3a4a; color: #00bcd4; }
.badge.failed { background: #4a1a1a; color: #f44336; }
.badge.pending { background: #2a2a1a; color: #ff9800; }
.error { color: #f44336; margin-bottom: 0.75rem; }
a { color: #7b9fff; }
</style>
