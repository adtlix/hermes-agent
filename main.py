import os
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from google import genai

app = FastAPI()

# API-Client initialisieren
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

class PromptRequest(BaseModel):
    prompt: str

def hermes_background_worker():
    """Hintergrund-Thread für autonome Aufgaben des Agenten."""
    print("[Hermes] Hintergrund-Dienst aktiv.")
    while True:
        # Hier können zeitgesteuerte Agenten-Tasks laufen
        time.sleep(60)

@app.on_event("startup")
def startup():
    threading.Thread(target=hermes_background_worker, daemon=True).start()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Hermes Cloud Agent",
        "api_configured": client is not None
    }

@app.post("/chat")
def generate_response(req: PromptRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY ist nicht konfiguriert.")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=req.prompt
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
