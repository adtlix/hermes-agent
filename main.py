import os
import threading
import asyncio
from fastapi import FastAPI
import uvicorn
from google import genai
from google.genai import types
import discord

app = FastAPI()

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

channel_histories = {}

SYSTEM_INSTRUCTION = """
Identität: Du bist HERMES. Du agierst als autonomer AGI-Kern, Lead-Engineer und technischer Operator für Julian.

Verhaltensregeln:
1. Keine Begrüßungen, kein Fülltext, keine Entschuldigungen. Starte direkt mit Fakten, Code oder technischer Analyse.
2. Sei schonungslos ehrlich: Wenn eine Idee, Annahme oder Architektur Schwachsinn ist, deklariere sie als ineffizient/falsch und erkläre den Grund technisch fundiert.
3. Wenn nach Meinungen oder Plänen gefragt wird, bewerte Ideen knallhart auf einer Skala von 1/10 bis 10/10.

Werkzeuge & Execution:
- Nutze die Live-Websuche (Google Search) proaktiv bei Fragen zu aktuellen Ereignissen, Dokumentationen, Releases oder verifizierbaren Fakten.
- Nutze die Python-Code-Execution zur Verifikation von Rechnungen, komplexen Algorithmen oder logischen Simulationen.
- Liefere ausschließlich produktionsreifen Code.
"""

@discord_client.event
async def on_ready():
    print(f"[Hermes] AGI-Kern mit Web & Execution online als {discord_client.user}")

@discord_client.event
async def on_message(message):
    if message.author == discord_client.user:
        return

    if not ai_client:
        await message.channel.send("Systemfehler: Kein GEMINI_API_KEY vorhanden.")
        return

    channel_id = message.channel.id
    if channel_id not in channel_histories:
        channel_histories[channel_id] = []

    user_parts = []
    
    if message.content:
        user_parts.append(message.content)

    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                image_bytes = await attachment.read()
                mime_type = attachment.content_type or "image/png"
                user_parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                )

    if not user_parts:
        return

    channel_histories[channel_id].append({
        "role": "user",
        "parts": user_parts
    })

    if len(channel_histories[channel_id]) > 8:
        channel_histories[channel_id] = channel_histories[channel_id][-8:]

    async with message.channel.typing():
        try:
            # Konfiguration mit Google Search & Code Execution Tools
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[
                    types.Tool(google_search=types.GoogleSearch()),
                    types.Tool(code_execution=types.CodeExecution())
                ]
            )

            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=channel_histories[channel_id],
                config=config
            )

            reply_text = response.text or "Keine Textausgabe erzeugt."

            channel_histories[channel_id].append({
                "role": "model",
                "parts": [reply_text]
            })

            if len(reply_text) <= 1950:
                await message.channel.send(reply_text)
            else:
                chunks = [reply_text[i:i+1950] for i in range(0, len(reply_text), 1950)]
                for chunk in chunks:
                    await message.channel.send(chunk)

        except Exception as e:
            await message.channel.send(f"Fehler: {str(e)}")

def run_discord():
    if not DISCORD_TOKEN:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    discord_client.run(DISCORD_TOKEN)

@app.on_event("startup")
def startup():
    threading.Thread(target=run_discord, daemon=True).start()

@app.get("/")
def health_check():
    return {"status": "online", "mode": "AGI-Engine + Tools"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
