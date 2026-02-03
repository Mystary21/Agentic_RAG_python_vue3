import json
import httpx
from typing import AsyncGenerator, Dict, List

class ResponseSynthesizer:
    def __init__(self, ollama_base_url: str, model_name: str = "llama3.1:8b-instruct-q5_K_M"):
        self.base_url = ollama_base_url.rstrip("/")
        self.default_model = model_name
        self.client = httpx.AsyncClient(timeout=300.0)

    async def generate_response_stream(self, user_query: str, tool_output: str, intent: str, chat_history: List[Dict], model_name: str = None) -> AsyncGenerator[str, None]:
        
        target_model = model_name if model_name else self.default_model
        
        system_instruction = "You are a helpful AI assistant."
        final_prompt = user_query
        
        if intent == "search" or intent == "vision_qa":
            system_instruction += " Use the provided context to answer."
            final_prompt = f"--- CONTEXT ---\n{tool_output}\n--- END CONTEXT ---\n\nQuestion: {user_query}"

        messages = [
            {"role": "system", "content": system_instruction},
            *chat_history[-5:],
            {"role": "user", "content": final_prompt}
        ]

        async with self.client.stream("POST", f"{self.base_url}/api/chat", json={
            "model": target_model, "messages": messages, "stream": True
        }) as response:
            async for line in response.aiter_lines():
                if not line: continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token: yield token
                except: continue