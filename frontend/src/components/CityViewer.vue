<template>
  <div class="city-viewer-wrapper">
    <canvas ref="canvas" class="city-canvas" />
    <div class="legend">
      <div v-for="(color, cat) in CATEGORY_COLORS" :key="cat" class="legend-item">
        <span class="dot" :style="{ background: color }" />
        {{ cat }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as THREE from 'three'

const props = defineProps({
  trajectories: { type: Array, default: () => [] },
  cityBounds: {
    type: Object,
    default: () => ({ lat_min: 35.0, lat_max: 36.0, lon_min: 139.0, lon_max: 140.0 }),
  },
})

const canvas = ref(null)

// Intent category → hex color
const CATEGORY_COLORS = {
  dining: '#ff6b6b',
  shopping: '#ffd93d',
  transport: '#6bcfff',
  residential: '#a8e063',
  park: '#56ab2f',
  office: '#4facfe',
  entertainment: '#f7797d',
  education: '#b06ab3',
  healthcare: '#ff9a9e',
  convenience: '#fddb92',
  other: '#aaaaaa',
}

let renderer, scene, camera, animFrameId
const agentMeshes = new Map()  // agent_id -> THREE.Mesh

function latLonToXZ(lat, lon) {
  const { lat_min, lat_max, lon_min, lon_max } = props.cityBounds
  const x = ((lon - lon_min) / (lon_max - lon_min) - 0.5) * 200
  const z = ((lat - lat_min) / (lat_max - lat_min) - 0.5) * 200
  return { x, z }
}

function initThree() {
  const w = canvas.value.clientWidth
  const h = canvas.value.clientHeight

  renderer = new THREE.WebGLRenderer({ canvas: canvas.value, antialias: true })
  renderer.setSize(w, h)
  renderer.setPixelRatio(window.devicePixelRatio)

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0a0a1a)

  camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000)
  camera.position.set(0, 150, 100)
  camera.lookAt(0, 0, 0)

  // Ground grid
  const gridHelper = new THREE.GridHelper(220, 40, 0x222266, 0x111133)
  scene.add(gridHelper)

  // Ambient + directional light
  scene.add(new THREE.AmbientLight(0xffffff, 0.6))
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
  dirLight.position.set(50, 100, 50)
  scene.add(dirLight)

  animate()
}

function animate() {
  animFrameId = requestAnimationFrame(animate)
  renderer.render(scene, camera)
}

function updateAgents(trajectories) {
  const seen = new Set()
  for (const rec of trajectories) {
    seen.add(rec.agent_id)
    const { x, z } = latLonToXZ(rec.lat, rec.lon)
    const color = CATEGORY_COLORS[rec.intent_category] || CATEGORY_COLORS.other
    if (agentMeshes.has(rec.agent_id)) {
      const mesh = agentMeshes.get(rec.agent_id)
      mesh.position.set(x, 1, z)
      mesh.material.color.set(color)
    } else {
      const geo = new THREE.SphereGeometry(1, 8, 8)
      const mat = new THREE.MeshLambertMaterial({ color })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(x, 1, z)
      scene.add(mesh)
      agentMeshes.set(rec.agent_id, mesh)
    }
  }
  // Remove agents no longer in the data
  for (const [id, mesh] of agentMeshes) {
    if (!seen.has(id)) {
      scene.remove(mesh)
      agentMeshes.delete(id)
    }
  }
}

watch(() => props.trajectories, (trajs) => updateAgents(trajs), { deep: true })

onMounted(() => {
  initThree()
  if (props.trajectories.length) updateAgents(props.trajectories)
})

onUnmounted(() => {
  cancelAnimationFrame(animFrameId)
  renderer?.dispose()
})
</script>

<style scoped>
.city-viewer-wrapper { position: relative; width: 100%; height: 520px; }
.city-canvas { width: 100%; height: 100%; display: block; }
.legend {
  position: absolute; top: 10px; right: 10px;
  background: rgba(10,10,26,0.85); border-radius: 6px;
  padding: 0.6rem 0.9rem; font-size: 0.78rem;
  display: flex; flex-direction: column; gap: 0.3rem;
}
.legend-item { display: flex; align-items: center; gap: 0.4rem; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
</style>
