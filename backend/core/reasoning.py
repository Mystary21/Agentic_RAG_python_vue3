import json
import httpx
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

# --- Data Structures (Schema Definition) ---

class IntentType(str, Enum):
    """
    Enumeration of possible user intents.
    """
    SEARCH = "search"           # Needs external info (RAG)
    SUMMARIZE = "summarize"     # Needs to process provided text
    CALCULATE = "calculate"     # Needs math/logic tool
    CHITCHAT = "chitchat"       # General conversation, no tools needed

class QueryAnalysis(BaseModel):
    """
    Structured output from the reasoning model.
    """
    intent: IntentType = Field(..., description="The classified intent of the user query.")
    key_entities: List[str] = Field(default_factory=list, description="Important entities extracted (e.g., company names, dates).")
    missing_info: Optional[str] = Field(None, description="If intent is unclear, what information is missing?")
    is_safe: bool = Field(True, description="Safety check filter.")

# --- The Core Engine ---

class ReasoningEngine:
    """
    The cognitive layer responsible for understanding user input before execution.
    Target Model: llama3.1:8b-instruct (Hosted on Server 230)
    """

    def __init__(self, ollama_base_url: str, model_name: str = "llama3.1:8b-instruct-q5_K_M"):
        """
        Initialize connection to the remote A100 server.
        
        Args:
            ollama_base_url: URL to your Server 230 (e.g., "http://192.168.1.230:11434")
            model_name: The specific model tag from your list.
        """
        self.base_url = ollama_base_url
        self.model = model_name
        self.client = httpx.AsyncClient(timeout=30.0) # Set reasonable timeout

    async def analyze_query(self, user_query: str, chat_history: List[Dict]) -> QueryAnalysis:
        """
        Main entry point: Classifies intent and parses parameters in a SINGLE call.
        
        Architecture Note:
        Instead of making two separate calls (one for intent, one for entities),
        we optimize latency by asking the LLM to return a single JSON object containing both.
        """
        
        # 1. Construct the System Prompt
        # We instruct the model to act as a strict JSON router.
        system_prompt = (
            "You are the Reasoning Engine of an advanced AI Agent. "
            "Your goal is to analyze the user's input and route it to the correct tool. "
            "You MUST output raw JSON only, adhering to the specified schema. "
            "Do not include markdown blocks or explanations."
            f"Valid Intents: {[e.value for e in IntentType]}"
        )

        # 2. Prepare the context (Current query + minimal history)
        # We limit history to prevent context overflow and focus on immediate intent.
        messages = [
            {"role": "system", "content": system_prompt},
            *chat_history[-3:], # Keep only last 3 turns for context
            {"role": "user", "content": f"Analyze this query: {user_query}"}
        ]

        # 3. Call Remote Ollama (Force JSON mode)
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "format": "json", # Critical: Forces Ollama to emit JSON
                    "stream": False,
                    "options": {
                        "temperature": 0.1, # Low temp for deterministic routing
                        "seed": 42
                    }
                }
            )
            response.raise_for_status()
            result_json = response.json()
            
            # 4. Parse and Validate
            content = result_json.get("message", {}).get("content", "{}")
            parsed_data = QueryAnalysis.model_validate_json(content)
            return parsed_data

        except Exception as e:
            # Fallback logic is crucial for production reliability
            print(f"[Error] Reasoning failed: {e}")
            # Default fallback: Treat as chitchat or ask for clarification
            return QueryAnalysis(intent=IntentType.CHITCHAT, key_entities=[])

    async def rewrite_query(self, raw_query: str, history: List[Dict]) -> str:
        """
        Reformulates the user query to be standalone and search-friendly.
        
        Use Case:
        User: "Who is the CEO of Nvidia?" -> Bot: "Jensen Huang."
        User: "How old is he?" -> Rewrite: "How old is Jensen Huang?"
        """
        
        prompt = (
            f"Given the following chat history and a follow-up question, "
            f"rephrase the follow-up question to be a standalone question. "
            f"Chat History: {history}\n"
            f"Follow-up: {raw_query}\n"
            f"Standalone Question:"
        )
        
        # Call Ollama (Standard text mode, not JSON)
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", raw_query).strip()
        return raw_query