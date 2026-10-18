import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  base: '/',
  server: { proxy: { '/api': 'http://127.0.0.1:7860' } },
  build: {
    emptyOutDir: false,
    license: { fileName: 'dependency-licenses.md' },
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/echarts') || id.includes('node_modules/zrender')) return 'charts'
          if (id.includes('node_modules')) return 'vendor'
        },
      },
    },
  },
})
