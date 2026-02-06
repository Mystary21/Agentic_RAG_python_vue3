# 進入前端補裝依賴
# Enter folder frontend to install the stuff of [vue3]
cd Agentic_RAG/frontend


npm install

# if meet some version problem
npm install -D vue-tsc@latest typescript@latest

npm run build

# use the docker to compile
docker run --rm -v $(pwd):/app -w /app node:18-alpine sh -c "npm install && npm run build"
