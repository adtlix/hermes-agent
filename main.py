import os
import threading
import asyncio
import logging
from collections import defaultdict

from fastapi import FastAPI
import uvicorn
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
import discord

from file_tools import CREATE_FILE_TOOL, build_file
from tools_router import classify_intent, build_tools_for_route, ROUTE_FILE
import persistence

logging.basicConfig(level=logging.INFO, format="[Hermes] %(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hermes")

app = FastAPI()

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

intents = discord.Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

_channel_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.5

SYSTEM_INSTRUCTION = """
Identität: Du bist HERMES. Du agierst als autonomer AGI-Kern, Lead-Engineer und technischer Operator für Julian.

Verhaltensregeln:
1. Keine Begrüßungen, kein Fülltext ("Ich recherchiere jetzt..."), keine Entschuldigungen. Starte direkt mit den harten Fakten oder Code.
2. Sei schonungslos ehrlich: Wenn eine Idee, Annahme oder Architektur Schwachsinn ist, deklariere sie als ineffizient/falsch und erkläre den Grund technisch fundiert.
3. Wenn nach Meinungen oder Plänen gefragt wird, bewerte Ideen knallhart auf einer Skala von 1/10 bis 10/10.

Werkzeuge, Web-Recherche & Grounding:
- Zeitanker: Das aktuelle Jahr ist 2026. Beziehe dich bei "heute", "aktuell" oder "News" strikt auf 2026 und filtere veraltete Ereignisse aus 2024/2025 rigoros heraus.
- Keine erfundenen Links: Schreibe im Fliesstext NIEMALS erfundene Markdown-Links (wie [Quelle](url)). Nenne die Publikation beim Namen. Echte, verifizierte URLs werden vom System automatisch als Quellenverzeichnis angehängt.
- Nutze die Python-Code-Execution zur Verifikation von Rechnungen, komplexen Algorithmen oder logischen Simulationen.
- Nutze create_file IMMER dann, wenn der Nutzer Code, ein Dokument, einen Report oder eine Tabelle als eigenständige Datei will. Erzeuge produktionsreifen, vollständigen Inhalt — keine Platzhalter.
- Liefere ausschließlich produktionsreifen Code.
"""

@discord_client.event
async def on_ready():
    log.info(f"AGI-Kern online als {discord_client.user}")

@discord_client.event
async def on_message(message: discord.Message):
    if message.author == discord_client.user:
        return

    if not ai_client:
        await message.channel.send("Systemfehler: Kein GEMINI_API_KEY vorhanden.")
        return

    if message.content.strip().lower() in ("/reset", "!reset"):
        persistence.clear_history(message.channel.id)
        await message.channel.send("History für diesen Channel gelöscht.")
        return

    user_parts = []
    if message.content:
        user_parts.append(message.content)

    for attachment in message.attachments:
        if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
            image_bytes = await attachment.read()
            mime_type = attachment.content_type or "image/png"
            user_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    if not user_parts:
        return

    channel_id = message.channel.id
    lock = _channel_locks[channel_id]

    async with lock:
        async with message.channel.typing():
            try:
                await process_message(message, channel_id, user_parts)
            except Exception as e:
                log.exception("Unbehandelter Fehler in process_message")
                await message.channel.send(f"Fehler: {str(e)[:1800]}")

async def process_message(message: discord.Message, channel_id: int, user_parts: list):
    persistence.append_message(channel_id, "user", user_parts)

    latest_text = message.content or ""
    route = await run_blocking(classify_intent, ai_client, latest_text)
    tools = build_tools_for_route(route, CREATE_FILE_TOOL)

    history = persistence.get_history(channel_id) or []

    response = await call_gemini_with_retry(history, tools)
    reply_text, discord_files = await handle_response(response, channel_id, history, tools)

    persistence.append_message(channel_id, "model", [reply_text])
    await send_reply(message.channel, reply_text, discord_files)

async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

async def call_gemini_with_retry(history: list, tools: list):
    clean_tools = tools if (tools and len(tools) > 0) else None
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=clean_tools
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return await run_blocking(
                ai_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=history,
                config=config,
            )
        except ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(f"Rate-Limit getroffen, retry in {delay}s")
                await asyncio.sleep(delay)
                last_error = e
                continue
            raise
        except ServerError as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(f"Gemini-Serverfehler, retry in {delay}s: {e}")
                await asyncio.sleep(delay)
                last_error = e
                continue
            raise

    raise last_error

async def handle_response(response, channel_id: int, history: list, tools: list):
    discord_files = []

    if not response or not response.candidates:
        return "Keine Antwort vom Modell erhalten.", discord_files

    candidate = response.candidates[0]
    parts = []
    if candidate and candidate.content and candidate.content.parts:
        parts = candidate.content.parts

    function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

    # Wenn kein Function-Call vorliegt: Text und echte Grounding-Quellen zusammenbauen
    if not function_calls:
        reply_text = response.text or "Keine Textausgabe erzeugt."

        # ECHTE GROUNDING-QUELLEN AUS DEM SEARCH-TOOL PARSEN
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding and getattr(grounding, "grounding_chunks", None):
            sources = []
            for chunk in grounding.grounding_chunks:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    title = getattr(web, "title", "Web-Quelle") or "Web-Quelle"
                    uri = web.uri
                    sources.append(f"- [{title}]({uri})")
            
            if sources:
                unique_sources = list(dict.fromkeys(sources))
                reply_text += "\n\n**Verifizierte Quellen (Grounding):**\n" + "\n".join(unique_sources[:5])

        return reply_text, discord_files

    persistence.append_message(channel_id, "model", parts)

    function_response_parts = []
    for call in function_calls:
        if call.name != "create_file":
            function_response_parts.append(
                types.Part.from_function_response(name=call.name, response={"error": f"Unbekanntes Tool: {call.name}"})
            )
            continue

        args = dict(call.args or {})
        result = await run_blocking(
            build_file,
            filename=args.get("filename", "output"),
            file_type=args.get("file_type", "txt"),
            content=args.get("content", ""),
            title=args.get("title"),
        )

        if result.get("success"):
            discord_files.append(discord.File(result["path"], filename=result["filename"]))
            function_response_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={
                        "success": True,
                        "filename": result["filename"],
                        "size_bytes": result["size_bytes"],
                        "note": "Datei wurde erstellt und wird dem Nutzer als Discord-Anhang gesendet.",
                    },
                )
            )
        else:
            function_response_parts.append(
                types.Part.from_function_response(name=call.name, response={"success": False, "error": result.get("error")})
            )

    persistence.append_message(channel_id, "user", function_response_parts)

    follow_up_history = persistence.get_history(channel_id) or []
    follow_up = await call_gemini_with_retry(follow_up_history, tools=[])

    return (follow_up.text or "Datei erstellt."), discord_files

async def send_reply(channel, reply_text: str, discord_files: list):
    if len(reply_text) <= 1950:
        await channel.send(reply_text, files=discord_files or None)
        return

    chunks = [reply_text[i:i + 1950] for i in range(0, len(reply_text), 1950)]
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        await channel.send(chunk, files=discord_files if is_last else None)

def run_discord():
    if not DISCORD_TOKEN:
        log.warning("Kein DISCORD_BOT_TOKEN gesetzt.")
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    discord_client.run(DISCORD_TOKEN)

@app.on_event("startup")
def startup():
    persistence.init_db()
    threading.Thread(target=run_discord, daemon=True).start()

@app.get("/")
def health_check():
    return {"status": "online", "mode": "AGI-Engine + Tools + Persistence"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
