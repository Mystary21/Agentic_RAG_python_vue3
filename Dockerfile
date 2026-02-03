# --- Stage 1: Build Frontend (Node 18) ---
# 這個階段完全在容器內運作，不受 Host Node 版本或 409 錯誤影響
FROM node:18-alpine as frontend-builder

WORKDIR /app/frontend

# Copy dependency definitions
COPY frontend/package.json frontend/package-lock.json* ./

# Clean install dependencies
# 使用 --legacy-peer-deps 避免某些版本衝突
RUN npm install --legacy-peer-deps && npm install -D vue-tsc@latest typescript@latest

# Copy source code
COPY frontend/ .

# Build Vue3
RUN npm run build

# --- Stage 2: Setup Backend (Python 3.11) ---
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend Code
COPY backend/ .

# Copy the built assets from Stage 1
# 這裡是最神奇的地方：我們從 Stage 1 複製編譯好的檔案，完全不需要在 Host 生成 dist
COPY --from=frontend-builder /app/frontend/dist ./static

# Environment Variables
ENV OLLAMA_HOST="http://10.199.1.230:8082/"
ENV PORT=8085

# Expose Port
EXPOSE 8085

# Run Application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8085"]
