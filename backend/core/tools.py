import chromadb
import httpx
import base64
from typing import List, Dict, Optional

class VectorStoreManager:
    """
    Manages the interaction with the Vector Database (ChromaDB) and 
    Embedding Model (Ollama).
    """
    def __init__(self, ollama_base_url: str, collection_name: str = "knowledge_base", embedding_model: str = "nomic-embed-text:latest"):
        self.base_url = ollama_base_url
        self.embedding_model = embedding_model
        # Persistent path inside container
        self.client = chromadb.PersistentClient(path="/app/chroma_db")
        # Ensure we use cosine distance for better text similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        self.http_client = httpx.AsyncClient(timeout=30.0)

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
        # Simple ID generation based on current count to avoid collision in basic usage
        current_count = self.collection.count()
        ids = [str(current_count + i) for i in range(len(documents))]
        
        embeddings = []
        for doc in documents:
            embeddings.append(await self._get_embedding(doc))
            
        self.collection.upsert(
            ids=ids, 
            documents=documents, 
            embeddings=embeddings, 
            metadatas=metadatas
        )
        print(f"[Info] Added {len(documents)} documents to vector store.")

class VisionTool:
    """
    Provides visual capabilities using Multimodal LLMs.
    """
    def __init__(self, ollama_base_url: str, model_name: str = "llama3.2-vision:latest"):
        self.base_url = ollama_base_url
        self.model = model_name
        self.client = httpx.AsyncClient(timeout=60.0)

    async def analyze_image(self, image_base64: str, prompt: str) -> str:
        try:
            # Cleanup header if present
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
