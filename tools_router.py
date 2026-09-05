"""
_ROUTER_INSTRUCTION = """
Du bist ein kompromissloser Intent-Klassifikator. Antworte AUSSCHLIESSLICH mit genau einem Wort:

web_code  -> wenn die Anfrage Recherche, aktuelle News, Ereignisse ("letzte Woche", "heute", "News", "Recherche"), Wetter, Faktenprüfung, Mathematik oder Code-Ausführung erfordert.
file      -> wenn die Anfrage eine herunterladbare Datei erfordert (z.B. "als PDF", "als Excel", "als .py Datei", "Download").
plain     -> NUR für reine Logikfragen, Code-Refactorings im Chat oder Begriffsdefinitionen ohne Aktualitätsbezug.

Regel: Sobald nach "News", "neu", "letzte Woche", "Recherche", "Quellen" oder aktuellen Fakten gefragt wird, MUSS die Antwort zwingend "web_code" lauten.
"""
"""

from google import genai
from google.genai import types

ROUTE_WEB_CODE = "web_code"
ROUTE_FILE = "file"
ROUTE_PLAIN = "plain"

_ROUTER_INSTRUCTION = """
Du bist ein reiner Klassifikator, kein Chat-Assistent. Antworte NUR mit
genau einem der folgenden Wörter, ohne Anführungszeichen, ohne Satzzeichen,
ohne Erklärung:

web_code  -> wenn die Anfrage aktuelle Web-Infos, Recherche, oder eine
             Berechnung/Code-Ausführung zur Verifikation braucht.
file      -> wenn die Anfrage eindeutig eine herunterladbare Datei will
             (Code als .py-Datei, PDF, Excel-Tabelle, Textdatei zum
             Speichern), NICHT nur Code als Chat-Antwort.
plain     -> für alles andere: normale Konversation, Meinungen, Erklärungen,
             Code-Snippets die nur im Chat gezeigt werden sollen.

Im Zweifel zwischen file und plain: wenn das Wort "Datei", "Download",
"als .py/.pdf/.xlsx", "schick mir eine Datei" o.ä. vorkommt -> file.
Sonst plain.
"""


def classify_intent(ai_client: genai.Client, latest_user_text: str) -> str:
    """Ein schneller, günstiger Call zur Tool-Auswahl. Fällt bei Fehlern sicher auf 'plain' zurück."""
    if not latest_user_text.strip():
        return ROUTE_PLAIN

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [latest_user_text]}],
            config=types.GenerateContentConfig(
                system_instruction=_ROUTER_INSTRUCTION,
                max_output_tokens=10,
                temperature=0,
            ),
        )
        route = (response.text or "").strip().lower()
        if route in (ROUTE_WEB_CODE, ROUTE_FILE, ROUTE_PLAIN):
            return route
        return ROUTE_PLAIN
    except Exception:
        # Router-Fehler dürfen den Hauptchat niemals blockieren.
        return ROUTE_PLAIN


def build_tools_for_route(route: str, file_tool) -> list:
    if route == ROUTE_WEB_CODE:
        return [{"google_search": {}}, {"code_execution": {}}]
    if route == ROUTE_FILE:
        return [file_tool]
    return []
