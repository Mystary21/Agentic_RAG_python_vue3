import json
import httpx
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class IntentType(str, Enum):
    SEARCH = "search"
    SUMMARIZE = "summarize"
    CALCULATE = "calculate"
    CHITCHAT = "chitchat"
    VISION_QA = "vision_qa"

class QueryAnalysis(BaseModel):
    intent: IntentType = Field(..., description="The classified intent.")
    key_entities: List[str] = Field(default_factory=list)
    missing_info: Optional[str] = Field(None)
    is_safe: bool = Field(True)

class ReasoningEngine:
    def __init__(self, ollama_base_url: str, model_name: str = "llama3.1:8b-instruct-q5_K_M"):
        self.base_url = ollama_base_url.rstrip("/")
        self.default_model = model_name # Keep default as backup
        self.client = httpx.AsyncClient(timeout=300.0)

    async def analyze_query(self, user_query: str, chat_history: List[Dict], has_image: bool = False, model_name: str = None) -> QueryAnalysis:
        if has_image:
            return QueryAnalysis(intent=IntentType.VISION_QA, key_entities=[])
        
        # Use provided model or fallback to default
        target_model = model_name if model_name else self.default_model

        system_prompt = (
            "You are the Reasoning Engine. Output raw JSON only. "
            f"Valid Intents: {[e.value for e in IntentType]}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *chat_history[-3:],
            {"role": "user", "content": f"Analyze: {user_query}"}
        ]
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": target_model,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1}
                }
            )
            response.raise_for_status()
            return QueryAnalysis.model_validate_json(response.json().get("message", {}).get("content", "{}"))
        except Exception as e:
            print(f"[Reasoning Error] Using model {target_model}. Details: {e}")
            return QueryAnalysis(intent=IntentType.CHITCHAT)

    async def rewrite_query(self, raw_query: str, history: List[Dict]) -> str:
        return raw_query