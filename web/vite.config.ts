import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 8732 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    rollupOptions: {
      input: { main: 'index.html', verify: 'verify.html' },
    },
  },
});
