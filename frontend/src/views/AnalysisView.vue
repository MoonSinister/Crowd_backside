<template>
  <div class="analysis-view">
    <h1>Intent & Emergence Analysis</h1>

    <div class="controls">
      <label>
        Run ID
        <input v-model.number="runId" type="number" min="1" placeholder="Run ID" />
      </label>
      <button @click="load">Analyse</button>
    </div>

    <section v-if="store.intentStats.length" class="card">
      <h2>Intent Distribution</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Total Moves</th>
            <th>Fast Path (experience reuse)</th>
            <th>Slow Path (LLM reasoning)</th>
            <th>Fast %</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="s in sortedStats" :key="s.intent_category">
            <td>{{ s.intent_category }}</td>
            <td>{{ s.count }}</td>
            <td>{{ s.fast_path_count }}</td>
            <td>{{ s.slow_path_count }}</td>
            <td>{{ pct(s.fast_path_count, s.count) }}</td>
          </tr>
        </tbody>
      </table>

      <!-- Simple bar chart via inline SVG -->
      <svg class="bar-chart" viewBox="0 0 600 220">
        <g v-for="(s, i) in sortedStats" :key="s.intent_category">
          <rect
            :x="i * (600 / sortedStats.length)"
            :y="220 - barHeight(s.count)"
            :width="600 / sortedStats.length - 4"
            :height="barHeight(s.count)"
            :fill="BAR_COLORS[i % BAR_COLORS.length]"
          />
          <text
            :x="i * (600 / sortedStats.length) + (600 / sortedStats.length - 4) / 2"
            y="218"
            text-anchor="middle"
            font-size="9"
            fill="#aaa"
          >{{ s.intent_category.slice(0, 6) }}</text>
        </g>
      </svg>
    </section>

    <p v-else class="hint">Enter a Run ID and click Analyse.</p>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useSimulationStore } from '@/stores/simulation'

const store = useSimulationStore()
const runId = ref(null)

const BAR_COLORS = [
  '#ff6b6b','#ffd93d','#6bcfff','#a8e063','#56ab2f',
  '#4facfe','#f7797d','#b06ab3','#ff9a9e','#fddb92',
]

async function load() {
  if (!runId.value) return
  await store.fetchIntentStats(runId.value)
}

const sortedStats = computed(() =>
  [...store.intentStats].sort((a, b) => b.count - a.count)
)
const maxCount = computed(() => Math.max(1, ...sortedStats.value.map((s) => s.count)))
const barHeight = (count) => Math.max(2, (count / maxCount.value) * 180)
const pct = (n, d) => d > 0 ? ((n / d) * 100).toFixed(1) + '%' : '0%'
</script>

<style scoped>
.analysis-view { padding: 1.5rem; max-width: 1100px; margin: 0 auto; }
h1 { margin-bottom: 1rem; }
.controls { display: flex; gap: 0.75rem; align-items: flex-end; margin-bottom: 1.5rem; }
label { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
input { background: #0a0a1a; border: 1px solid #3a3a6a; border-radius: 4px;
        color: #e0e0e0; padding: 0.4rem 0.6rem; width: 120px; }
button { background: #3a5cff; color: #fff; border: none; border-radius: 4px;
         padding: 0.5rem 1.2rem; cursor: pointer; }
.card { background: #12122a; border: 1px solid #2a2a4a; border-radius: 8px;
        padding: 1.25rem; margin-bottom: 1.5rem; }
h2 { margin-bottom: 1rem; font-size: 1.1rem; color: #aac4ff; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 1rem; }
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #2a2a4a; }
th { color: #7b9fff; }
.bar-chart { width: 100%; height: 230px; margin-top: 0.5rem; }
.hint { color: #666; margin-top: 2rem; text-align: center; }
</style>
