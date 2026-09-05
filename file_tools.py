"""
file_tools.py
--------------
Gibt Hermes die Fähigkeit, echte Dateien (py, txt, md, pdf, xlsx) selbst
zu erzeugen. Wird als Gemini Function-Calling-Tool eingebunden: das Modell
entscheidet selbst, wann eine Datei sinnvoller ist als reiner Chat-Text.

Ablauf:
1. main.py deklariert `CREATE_FILE_DECLARATION` als Tool bei Gemini.
2. Antwortet Gemini mit einem function_call "create_file", ruft main.py
   `build_file(...)` hier auf.
3. Die erzeugte Datei liegt danach unter OUTPUT_DIR und kann 1:1 als
   discord.File hochgeladen werden.
"""

import os
import uuid
import json
from google.genai import types

OUTPUT_DIR = "/tmp/hermes_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_TYPES = ["py", "txt", "md", "pdf", "xlsx"]

# Discords normales Upload-Limit ohne Server-Boost/Nitro liegt bei 8 MB.
# Etwas Puffer nach unten, um Grenzfälle bei PDF/XLSX-Overhead abzufangen.
MAX_FILE_SIZE_BYTES = 7 * 1024 * 1024

# ---------------------------------------------------------------------------
# Tool-Deklaration für Gemini function calling
# ---------------------------------------------------------------------------
CREATE_FILE_DECLARATION = types.FunctionDeclaration(
    name="create_file",
    description=(
        "Erzeugt eine echte, herunterladbare Datei (Python-Skript, Text, "
        "Markdown, PDF oder Excel-Tabelle) und lädt sie direkt in den "
        "Discord-Chat hoch. Nutze dieses Tool immer dann, wenn der Nutzer "
        "Code, ein Dokument, einen Report oder eine Tabelle als Datei "
        "haben möchte, statt es nur als Chat-Text zu zeigen."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "filename": types.Schema(
                type="STRING",
                description="Dateiname ohne Pfad, z.B. 'analyse.py' oder 'report.pdf'. "
                             "Die Endung MUSS zum file_type passen.",
            ),
            "file_type": types.Schema(
                type="STRING",
                enum=SUPPORTED_TYPES,
                description="Dateityp: py, txt, md, pdf oder xlsx.",
            ),
            "content": types.Schema(
                type="STRING",
                description=(
                    "Für py/txt/md: der vollständige Rohtext der Datei. "
                    "Für pdf: der Inhalt als Markdown-ähnlicher Text "
                    "(Zeilen mit '# ' werden als Überschrift gerendert, "
                    "sonst als Absatz). "
                    "Für xlsx: JSON-String mit einer Liste von Zeilen, "
                    "z.B. '[[\"Name\",\"Preis\"],[\"Kaffee\",3.5]]'. "
                    "Die erste Zeile wird als Header fett formatiert."
                ),
            ),
            "title": types.Schema(
                type="STRING",
                description="Optionaler Titel, nur für PDF relevant (wird oben auf Seite 1 gesetzt).",
            ),
        },
        required=["filename", "file_type", "content"],
    ),
)

CREATE_FILE_TOOL = types.Tool(function_declarations=[CREATE_FILE_DECLARATION])


def _safe_path(filename: str) -> str:
    """Verhindert Path-Traversal und erzwingt einen eindeutigen Dateinamen."""
    base = os.path.basename(filename)
    unique_prefix = uuid.uuid4().hex[:8]
    return os.path.join(OUTPUT_DIR, f"{unique_prefix}_{base}")


def _build_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _build_pdf_file(path: str, content: str, title: str | None) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []

    if title:
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 0.5 * cm))

    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 0.3 * cm))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        else:
            # Minimal escaping für XML-Sonderzeichen in reportlab
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, styles["Normal"]))

    doc.build(story)


def _build_xlsx_file(path: str, content: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    try:
        rows = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"content ist kein gültiges JSON für xlsx: {e}")

    if not isinstance(rows, list) or not rows:
        raise ValueError("content muss eine nicht-leere Liste von Zeilen sein")

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True)

    # Grobe Auto-Breite
    for col_cells in ws.columns:
        length = max((len(str(c.value)) for c in col_cells if c.value is not None), default=10)
        ws.column_dimensions[col_cells[0].column_letter].width = min(length + 4, 50)

    wb.save(path)


def build_file(filename: str, file_type: str, content: str, title: str | None = None) -> dict:
    """
    Baut die Datei physisch auf der Platte und gibt Metadaten zurück.
    Wird von main.py aus dem function_call-Handler aufgerufen.
    """
    if file_type not in SUPPORTED_TYPES:
        return {"success": False, "error": f"Nicht unterstützter file_type: {file_type}"}

    if not filename.lower().endswith(f".{file_type}"):
        filename = f"{filename}.{file_type}"

    path = _safe_path(filename)

    try:
        if file_type in ("py", "txt", "md"):
            _build_text_file(path, content)
        elif file_type == "pdf":
            _build_pdf_file(path, content, title)
        elif file_type == "xlsx":
            _build_xlsx_file(path, content)
    except Exception as e:
        return {"success": False, "error": str(e)}

    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE_BYTES:
        os.remove(path)
        return {
            "success": False,
            "error": (
                f"Datei wäre {size / (1024*1024):.1f} MB groß, "
                f"Discord-Limit liegt bei ~7 MB. Inhalt kürzen oder aufteilen."
            ),
        }

    return {
        "success": True,
        "path": path,
        "filename": os.path.basename(path),
        "file_type": file_type,
        "size_bytes": size,
    }
