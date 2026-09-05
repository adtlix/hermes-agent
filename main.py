import os
import threading
import time
from fastapi import FastAPI
import uvicorn

app = FastAPI()

def hermes_agent_worker():
    """Hier läuft die Hintergrund-Logik deines Hermes-Agenten."""
    print("[Hermes] Agent erfolgreich im Hintergrund gestartet.")
    while True:
        # Platzhalter: Hier arbeitet dein Bot oder deine Task-Queue
        time.sleep(15)

@app.on_event("startup")
def on_startup():
    thread = threading.Thread(target=hermes_agent_worker, daemon=True)
    thread.start()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Hermes Agent",
        "uptime": "24/7 active"
    }

if __name__ == "__main__":
    # Render weist automatisch einen dynamischen Port über die Umgebungsvariable $PORT zu
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
