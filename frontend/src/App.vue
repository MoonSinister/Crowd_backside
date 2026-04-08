<template>
  <div id="root">
    <nav class="navbar">
      <span class="brand">🌆 Crowd Backside</span>
      <router-link to="/">Dashboard</router-link>
      <router-link to="/simulation">Simulation</router-link>
      <router-link to="/analysis">Analysis</router-link>
    </nav>
    <router-view />
  </div>
</template>

<style>
.navbar {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 0.75rem 1.5rem;
  background: #12122a;
  border-bottom: 1px solid #2a2a4a;
}
.brand { font-weight: bold; font-size: 1.1rem; color: #7b9fff; margin-right: auto; }
a { color: #aac4ff; text-decoration: none; }
a.router-link-active { color: #fff; font-weight: bold; }
</style>
