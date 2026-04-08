import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from './views/Dashboard.vue'
import SimulationView from './views/SimulationView.vue'
import AnalysisView from './views/AnalysisView.vue'

const routes = [
  { path: '/', component: Dashboard },
  { path: '/simulation', component: SimulationView },
  { path: '/analysis', component: AnalysisView },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
