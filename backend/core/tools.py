import chromadb
import httpx
import re
import base64
from typing import List, Dict, Optional

class VectorStoreManager:
    """
    Manages the interaction with the Vector Database (ChromaDB) and 
    Embedding Model (Ollama).
    Includes Chunking logic to handle large documents.
    """
    def __init__(self, ollama_base_url: str, collection_name: str = "knowledge_base", embedding_model: str = "nomic-embed-text:latest"):
        self.base_url = ollama_base_url.rstrip("/")
        self.embedding_model = embedding_model
        # Persistent path inside container
        self.client = chromadb.PersistentClient(path="/app/chroma_db")
        # Ensure we use cosine distance for better text similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        self.http_client = httpx.AsyncClient(timeout=300.0)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Splits text into smaller chunks with overlap to maintain context.
        """
        if not text: return []
        text = re.sub(r'\n+', '\n', text)
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= end: start = end
                
        return chunks

    async def _get_embedding(self, text: str) -> List[float]:
        try:
            response = await self.http_client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"[Embedding Error] {e}")
            return []

    async def search(self, query: str, top_k: int = 3) -> str:
        query_vec = await self._get_embedding(query)
        if not query_vec: return "Error generating embeddings."
        
        results = self.collection.query(
            query_embeddings=[query_vec], 
            n_results=top_k
        )
        
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not docs: return "No relevant info found in knowledge base."
        
        formatted_results = []
        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            source = meta.get("source", "Unknown") if meta else "Unknown"
            formatted_results.append(f"[Result {i+1}] (Source: {source}):\n{doc}")
            
        return "\n\n".join(formatted_results)

    async def add_documents(self, documents: List[str], metadatas: List[Dict]):
        all_chunks = []
        all_metadatas = []
        all_ids = []
        current_count = self.collection.count()
        
        print(f"[Ingest] Processing {len(documents)} documents...")

        for idx, doc in enumerate(documents):
            chunks = self._chunk_text(doc)
            original_meta = metadatas[idx] if idx < len(metadatas) else {}
            print(f"[Ingest] Document {idx+1} split into {len(chunks)} chunks.")

            for chunk_i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append(original_meta)
                all_ids.append(f"doc_{current_count + idx}_chunk_{chunk_i}")

        embeddings = []
        for i, chunk in enumerate(all_chunks):
            if i % 10 == 0: print(f"[Ingest] Embedding chunk {i+1}/{len(all_chunks)}...")
            emb = await self._get_embedding(chunk)
            if emb: embeddings.append(emb)

        if not embeddings:
            print("[Error] No embeddings generated.")
            return

        batch_size = 100
        total_chunks = len(embeddings) # Use embeddings length to be safe
        
        for i in range(0, total_chunks, batch_size):
            end = min(i + batch_size, total_chunks)
            self.collection.upsert(
                ids=all_ids[i:end], 
                documents=all_chunks[i:end], 
                embeddings=embeddings[i:end], 
                metadatas=all_metadatas[i:end]
            )
        print(f"[Info] Successfully added {total_chunks} chunks to vector store.")

class VisionTool:
    """
    Provides visual capabilities using Multimodal LLMs.
    """
    def __init__(self, ollama_base_url: str, model_name: str = "llama3.2-vision:latest"):
        self.base_url = ollama_base_url.rstrip("/")
        self.model = model_name
        self.client = httpx.AsyncClient(timeout=300.0)

    async def analyze_image(self, image_base64: str, prompt: str) -> str:
        try:
            if "," in image_base64: 
                image_base64 = image_base64.split(",")[1]
                
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user", 
                        "content": prompt, 
                        "images": [image_base64]
                    }],
                    "stream": False
                }
            )
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            print(f"[Vision Error] {e}")
            return "Error analyzing image."