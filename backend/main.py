import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Import custom modules
from core.reasoning import ReasoningEngine, IntentType
from core.tools import VectorStoreManager, VisionTool
from core.synthesizer import ResponseSynthesizer

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://10.199.1.230:8082")

# --- FastAPI Setup ---
app = FastAPI(title="Agentic RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Instances ---
reasoning_engine = None
vector_store = None
vision_tool = None
synthesizer = None

# --- API Models (必須放在 Endpoints 之前!) ---

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    image_data: Optional[str] = Field(None, description="Base64 encoded image string")

class IngestRequest(BaseModel):
    text_content: str
    metadata: Dict[str, Any]

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    global reasoning_engine, vector_store, vision_tool, synthesizer
    print(f"[System] Connecting to Ollama at {OLLAMA_BASE_URL}...")
    
    reasoning_engine = ReasoningEngine(ollama_base_url=OLLAMA_BASE_URL)
    vector_store = VectorStoreManager(ollama_base_url=OLLAMA_BASE_URL)
    vision_tool = VisionTool(ollama_base_url=OLLAMA_BASE_URL)
    synthesizer = ResponseSynthesizer(ollama_base_url=OLLAMA_BASE_URL)
    
    print("[System] All modules initialized successfully.")

# --- Endpoints ---

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        user_query = request.query
        history = request.history
        image_data = request.image_data
        
        # Phase 1: Reasoning
        if image_data:
            intent = IntentType.VISION_QA
            print(f"[Log] Intent detected: {intent} (Image Uploaded)")
        else:
            analysis = await reasoning_engine.analyze_query(user_query, history)
            intent = analysis.intent
            print(f"[Log] Intent detected: {intent}")

        # Phase 2: Tool Execution
        tool_output = ""
        if intent == IntentType.VISION_QA and image_data:
            tool_output = await vision_tool.analyze_image(image_data, prompt=user_query)
            
        elif intent == IntentType.SEARCH:
            tool_output = await vector_store.search(user_query, top_k=3)
            
        elif intent == IntentType.CALCULATE:
             tool_output = "Calculator tool not yet implemented."

        # Phase 3: Aggregation (Streaming)
        return StreamingResponse(
            synthesizer.generate_response_stream(
                user_query=user_query,
                tool_output=tool_output,
                intent=intent,
                chat_history=history
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        print(f"[Error] Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    try:
        await vector_store.add_documents(
            documents=[request.text_content],
            metadatas=[request.metadata]
        )
        return {"status": "success", "message": "Document indexed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Static Files & Frontend Serving ---

# Mount 'static' folder (Vue build output) to /assets
if os.path.exists("static"):
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

# Catch-all route for SPA (Vue Router)
@app.get("/{catchall:path}")
async def read_index(catchall: str):
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"error": "Frontend not built or static files missing"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8085, reload=True)
