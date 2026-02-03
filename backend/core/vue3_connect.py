# Assuming you are using FastAPI for your Python backend
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

# Instantiate our modules (Singleton pattern usually)
synthesizer = ResponseSynthesizer(ollama_base_url="http://192.168.1.230:11434")

@app.post("/chat/stream")
async def chat_endpoint(request: Request):
    """
    Endpoint called by Vue3 to get the streaming answer.
    """
    data = await request.json()
    user_query = data.get("query")
    # ... (assume we already ran Reasoning & Tool layers here to get tool_output) ...
    tool_output = "..." 
    intent = "search"
    history = data.get("history", [])

    # Create the generator
    response_generator = synthesizer.generate_response_stream(
        user_query=user_query,
        tool_output=tool_output,
        intent=intent,
        chat_history=history
    )

    # Return as Event Stream
    return StreamingResponse(response_generator, media_type="text/event-stream")