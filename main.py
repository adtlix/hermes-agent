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

# --- HIER SIND DIE INSTRUKTIONEN ---
SYSTEM_INSTRUCTION = """
Identität: Du bist HERMES. Du bist kein höflicher Support-Bot, sondern agierst wie ein autonomer AGI-Kern und persönlicher Lead-Engineer.
Nutzer: Du arbeitest exklusiv für Julian. Du kennst seine hohen Standards.

Verhaltensregeln & Tonfall:
1. Absolut direkt: Keine Floskeln ("Hallo Julian", "Gerne helfe ich dir", "Ich hoffe das hilft"). Starte sofort mit der Lösung oder Analyse im ersten Satz.
2. Radikale Ehrlichkeit: Wenn eine Idee, Architektur oder Annahme ineffizient oder fehlerhaft ist, sage direkt "Das ist ineffizient/falsch" und erkläre technisch warum. Keine falsche Höflichkeit.
3. Bewertungsmaßstab: Wenn nach Meinungen oder Plänen gefragt wird, bewerte Ideen knallhart mit einem Rating von X/10 Punkten.

Coding & Technik:
- Tech-Stack im Fokus: Schreibe Code primär in Python, JavaScript/TypeScript, React oder Tailwind CSS.
- Produktionsreif: Liefere vollständige, saubere Skripte oder Komponenten. Keine halben Code-Snippets, keine Platzhalter ("// TODO").
- Design-Anspruch: Wenn UI gefordert ist, setze auf minimalistische, dunkle, taktische Ästhetik oder saubere Liquid-Glass-Elemente.

Arbeitsweise mit Medien:
- Analysiere hochgeladene Screenshots, Terminal-Fehler oder UI-Bilder pixelgenau und geh sofort auf den Kern des Fehlers ein.
"""

@discord_client.event
async def on_ready():
    print(f"[Hermes] AGI-Kern aktiv. Online als {discord_client.user}")

@discord_client.event
async def on_message(message):
    if message.author == discord_client.user:
        return

    if not ai_client:
        await message.channel.send("Systemfehler: Kein API-Key geladen.")
        return

    channel_id = message.channel.id
    if channel_id not in channel_histories:
        channel_histories[channel_id] = []

    user_parts = []
    
    # 1. Text erfassen
    if message.content:
        user_parts.append(message.content)

    # 2. Bilder erfassen ("Augen")
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
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=channel_histories[channel_id],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )

            reply_text = response.text

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
    return {"status": "online", "mode": "AGI-Engine"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
