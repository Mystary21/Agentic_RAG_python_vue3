import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    // 讓本地開發 (npm run dev) 也能連到後端
    proxy: {
      '/chat': {
        target: 'http://localhost:8085', // 指向您的 Python FastAPI Port
        changeOrigin: true,
      }
    }
  }
})
