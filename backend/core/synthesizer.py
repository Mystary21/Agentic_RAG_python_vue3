import json
import httpx
from typing import AsyncGenerator, Dict, List, Any

class ResponseSynthesizer:
    """
    The final layer responsible for synthesizing the answer.
    It combines the retrieved context with the user query and streams the output.
    """

    def __init__(self, ollama_base_url: str, model_name: str = "llama3.1:8b-instruct-q5_K_M"):
        self.base_url = ollama_base_url
        self.model = model_name
        self.client = httpx.AsyncClient(timeout=60.0) # Longer timeout for generation

    async def generate_response_stream(self, 
                                       user_query: str, 
                                       tool_output: str, 
                                       intent: str,
                                       chat_history: List[Dict]) -> AsyncGenerator[str, None]:
        """
        Generates a streaming response from the LLM.
        
        Args:
            user_query: The original question from the user.
            tool_output: The raw text returned by the Search Tool (or other tools).
            intent: The classified intent (e.g., 'search', 'chitchat').
            chat_history: Previous conversation context.
            
        Yields:
            str: Chunks of the generated text (tokens).
        """
        
        # 1. Construct the System Prompt based on Intent
        # Architecture Note: Dynamic prompting helps the model adjust its tone.
        if intent == "search":
            system_instruction = (
                "You are a precise and helpful AI assistant. "
                "You have been provided with external information in the 'Context' section. "
                "Answer the user's question based MAINLY on that context. "
                "If the context contains the answer, cite the source (e.g., [Source: Nvidia 2023 Report]). "
                "If the context does not answer the question, admit it politely."
            )
            final_prompt = (
                f"--- CONTEXT START ---\n{tool_output}\n--- CONTEXT END ---\n\n"
                f"User Question: {user_query}"
            )
        else:
            # Fallback for chitchat
            system_instruction = "You are a helpful and friendly AI assistant."
            final_prompt = user_query

        # 2. Build Message Payload
        messages = [
            {"role": "system", "content": system_instruction},
            *chat_history[-5:], # Keep last 5 turns for conversational flow
            {"role": "user", "content": final_prompt}
        ]

        # 3. Stream Request to Ollama
        # We use a POST request with stream=True
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,  # Critical for real-time feedback
                "options": {
                    "temperature": 0.7, # Slightly higher temp for natural language generation
                }
            }
        ) as response:
            
            # 4. Process the Stream
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                try:
                    # Ollama sends JSON objects in the stream
                    chunk_json = json.loads(line)
                    
                    # Extract the token. 
                    # Note: 'message' structure differs slightly between /api/chat and /api/generate
                    # For /api/chat, it's usually inside 'message' -> 'content'
                    token = chunk_json.get("message", {}).get("content", "")
                    
                    if chunk_json.get("done", False):
                        break
                        
                    if token:
                        yield token
                        
                except json.JSONDecodeError:
                    continue