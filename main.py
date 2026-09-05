import os
import threading
import asyncio
from fastapi import FastAPI
import uvicorn
from google import genai
import discord

app = FastAPI()

# Tokens aus Umgebungsvariablen
DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini Client
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Discord Client initialisieren
intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

@discord_client.event
async def on_ready():
    print(f"[Hermes] Erfolgreich eingeloggt als {discord_client.user}")

@discord_client.event
async def on_message(message):
    # Eigene Nachrichten ignorieren
    if message.author == discord_client.user:
        return

    if not ai_client:
        await message.channel.send("Fehler: GEMINI_API_KEY fehlt auf dem Server.")
        return

    async with message.channel.typing():
        try:
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=message.content
            )
            reply_text = response.text
            # Discord Nachrichtenlimit von 2000 Zeichen beachten
            if len(reply_text) <= 2000:
                await message.channel.send(reply_text)
            else:
                for i in range(0, len(reply_text), 2000):
                    await message.channel.send(reply_text[i:i+2000])
        except Exception as e:
            await message.channel.send(f"Fehler: {str(e)}")

def run_discord():
    if not DISCORD_TOKEN:
        print("[Hermes] WARNUNG: DISCORD_BOT_TOKEN fehlt in Environment Variables.")
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    discord_client.run(DISCORD_TOKEN)

@app.on_event("startup")
def startup():
    threading.Thread(target=run_discord, daemon=True).start()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Hermes Discord Bot",
        "bot_configured": DISCORD_TOKEN is not None
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
