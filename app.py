

import os
import re
import textwrap
from io import BytesIO

from flask import Flask, render_template, request, send_file, session
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-flash-latest"

SECTION_TITLES = [
    ("topic", "Topic"),
    ("definition", "Definition"),
    ("elaborate", "Elaboration"),
    ("key points", "Key points"),
    ("examples", "Examples"),
    ("real", "Real-world usage"),
    ("mcq", "MCQs"),
    ("question and answer", "Question & answer (2 marks)"),
    ("long answer", "Long answer (10 marks)"),
    ("interview", "Interview preparation questions"),
    ("common mistake", "Common mistakes"),
    ("quick notes", "Quick notes"),
]


def build_prompt(topic: str) -> str:
    return (
        "You are an academic assistant.\n"
        f"Generate clear, revisable study notes for the topic: {topic}\n\n"
        "Output style rules:\n"
        "- Use plain text only\n"
        "- Do not use markdown symbols like **, *, #, ```\n"
        "- Keep writing simple, clean, and exam-friendly\n\n"
        "Include the following sections in this exact order:\n"
        "1. Topic\n"
        "2. Definition\n"
        "3. Elaborate about the topic\n"
        "4. Key points\n"
        "5. Examples\n"
        "6. How it is really used in real-time\n"
        "7. MCQs with answers\n"
        "8. Question and answer (2 marks)\n"
        "9. Long answer (10 marks)\n"
        "10. Interview preparation questions\n"
        "11. Common mistakes about the topic\n"
        "12. Quick notes about the topic\n"
    )


def sanitize_notes_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    cleaned_lines: list[str] = []
    for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.replace("```", "")
        line = line.replace("**", "")
        line = line.replace("*", "")
        line = line.replace("#", "")
        line = line.replace("`", "")
        line = line.replace("•", "-")
        line = re.sub(r"^\s*[-=]{3,}\s*$", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def find_section_title(line: str) -> str | None:
    normalized = line.strip().lower()
    normalized = re.sub(r"^\s*\d+[\.)]\s*", "", normalized)
    for key, title in SECTION_TITLES:
        if key in normalized:
            return title
    return None


def extract_heading_remainder(line: str) -> str:
    cleaned = re.sub(r"^\s*\d+[\.)]\s*", "", line).strip()
    if ":" in cleaned:
        return cleaned.split(":", 1)[1].strip()
    return ""


def classify_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    paragraphs: list[str] = []
    list_items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_list = bool(re.match(r"^(?:-|\*|•|\d+[\.)]|[a-dA-D][\.)])\s+", stripped))
        if is_list:
            text = re.sub(r"^(?:-|\*|•|\d+[\.)]|[a-dA-D][\.)])\s+", "", stripped)
            list_items.append(text)
        else:
            paragraphs.append(stripped)
    return paragraphs, list_items


def parse_notes(raw_notes: str) -> list[dict[str, list[str] | str]]:
    if not raw_notes:
        return []

    sections: list[dict[str, list[str] | str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in raw_notes.splitlines():
        if not line.strip():
            continue
        found_title = find_section_title(line)
        if found_title:
            if current_title:
                paragraphs, list_items = classify_lines(current_lines)
                sections.append(
                    {
                        "title": current_title,
                        "paragraphs": paragraphs,
                        "list_items": list_items,
                    }
                )
            current_title = found_title
            current_lines = []
            remainder = extract_heading_remainder(line)
            if remainder:
                current_lines.append(remainder)
        elif current_title:
            current_lines.append(line)

    if current_title:
        paragraphs, list_items = classify_lines(current_lines)
        sections.append(
            {"title": current_title, "paragraphs": paragraphs, "list_items": list_items}
        )

    return sections


def build_pdf_lines(topic: str, raw_notes: str) -> list[str]:
    lines: list[str] = []
    if topic:
        lines.append(f"Topic: {topic}")
        lines.append("")

    sections = parse_notes(raw_notes)
    if sections:
        for section in sections:
            lines.append(section["title"])
            for paragraph in section["paragraphs"]:
                lines.extend(textwrap.wrap(paragraph, width=90) or [""])
            for item in section["list_items"]:
                lines.extend(textwrap.wrap(f"- {item}", width=90) or [""])
            lines.append("")
    else:
        for line in raw_notes.splitlines():
            lines.extend(textwrap.wrap(line, width=90) or [""])

    return lines


def generate_pdf(topic: str, raw_notes: str) -> BytesIO:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 72
    y = height - 72

    for line in build_pdf_lines(topic, raw_notes):
        if y < 72:
            pdf.showPage()
            y = height - 72
        pdf.drawString(x_margin, y, line)
        y -= 14

    pdf.save()
    buffer.seek(0)
    return buffer


def build_chat_prompt(topic: str, history: list[dict[str, str]], question: str) -> str:
    lines = [
        "You are an academic assistant.",
        f"The student is studying: {topic}.",
        "Answer clearly and concisely.",
        "",
        "Conversation:",
    ]
    for message in history:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {message.get('content', '')}")
    lines.append(f"User: {question}")
    return "\n".join(lines)


@app.route("/", methods=["GET", "POST"])
def index():
    notes = session.get("notes", "")
    topic = session.get("topic", "")
    sections = parse_notes(notes)
    error = ""
    chat_error = ""
    chat_history = session.get("chat_history", [])

    if request.method == "POST":
        topic = request.form.get("topic", "").strip()
        if not topic:
            error = "Please enter a topic."
        elif not API_KEY:
            error = "Missing GEMINI_API_KEY in .env."
        else:
            try:
                genai.configure(api_key=API_KEY)
                model = genai.GenerativeModel(MODEL_NAME)
                response = model.generate_content(build_prompt(topic))
                notes = sanitize_notes_text(response.text or "")
                sections = parse_notes(notes)
                session["notes"] = notes
                session["topic"] = topic
                session["chat_history"] = []
            except Exception as exc:
                error = f"Failed to generate notes: {exc}"

    return render_template(
        "index.html",
        notes=notes,
        sections=sections,
        topic=topic,
        error=error,
        chat_error=chat_error,
        chat_history=chat_history,
    )


@app.route("/download", methods=["POST"])
def download():
    topic = request.form.get("topic", "").strip()
    notes = request.form.get("notes", "").strip()
    if not notes:
        return "No notes available to export.", 400

    safe_topic = re.sub(r"[^a-zA-Z0-9_-]+", "-", topic or "study-notes").strip("-")
    filename = f"{safe_topic or 'study-notes'}.pdf"
    pdf_buffer = generate_pdf(topic, notes)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/chat", methods=["POST"])
def chat():
    topic = request.form.get("topic", "").strip() or session.get("topic", "")
    question = request.form.get("question", "").strip()
    notes = session.get("notes", "")
    sections = parse_notes(notes)
    chat_history = session.get("chat_history", [])
    error = ""
    chat_error = ""

    if not question:
        chat_error = "Please enter a question."
    elif not API_KEY:
        chat_error = "Missing GEMINI_API_KEY in .env."
    elif not topic:
        chat_error = "Please generate notes first so I know the topic."
    else:
        try:
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel(MODEL_NAME)
            prompt = build_chat_prompt(topic, chat_history[-6:], question)
            response = model.generate_content(prompt)
            answer = response.text or ""
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
            session["chat_history"] = chat_history
        except Exception as exc:
            chat_error = f"Failed to answer: {exc}"

    return render_template(
        "index.html",
        notes=notes,
        sections=sections,
        topic=topic,
        error=error,
        chat_error=chat_error,
        chat_history=chat_history,
    )


if __name__ == "__main__":
    app.run(debug=True)
