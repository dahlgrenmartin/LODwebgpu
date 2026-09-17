import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub project pages serve from /<repo>/. Set VITE_BASE at build time,
  // e.g. VITE_BASE=/LODwebgpu/ npm run build.
  base: process.env.VITE_BASE ?? '/',
  server: { port: 8732 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    rollupOptions: {
      input: { main: 'index.html', verify: 'verify.html', spike: 'spike.html', kernels: 'kernels.html', graph: 'graph.html' },
    },
  },
});
