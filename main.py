"""
AI Chatbot with RAG — FastAPI Backend using Groq (Free)
Run: uvicorn main:app --reload --port 8080
"""
from dotenv import load_dotenv
load_dotenv()
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from groq import Groq

from rag_engine import add_document_to_index, retrieve_context, clear_index, UPLOAD_DIR
from domain_config import get_system_prompt
from interviewer import (
    parse_resume_and_generate_questions,
    generate_coding_questions,
    evaluate_answer,
    evaluate_code,
    generate_final_report
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="GenAI RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")
templates = Jinja2Templates(directory="templates")

client = Groq(api_key=GROQ_API_KEY)


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    domain: str = "general"
    use_rag: bool = True

class InterviewStart(BaseModel):
    resume_text: str
    job_role: str

class AnswerEval(BaseModel):
    question: str
    answer: str
    job_role: str

class CodeEval(BaseModel):
    problem: str
    code: str
    language: str

class FinalReport(BaseModel):
    job_role: str
    behavioral_scores: list
    coding_scores: list
    candidate_name: str = "Candidate"

class CodingQuestions(BaseModel):
    job_role: str
    language: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )

    rag_context = retrieve_context(last_user_msg) if req.use_rag else None
    system_prompt = get_system_prompt(req.domain, rag_context)

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages += [{"role": m.role, "content": m.content} for m in req.messages]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=api_messages,
        max_tokens=1024,
        temperature=0.7,
    )

    reply = response.choices[0].message.content
    return {"reply": reply, "rag_used": rag_context is not None}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"
        )

    # Clear old index before adding new document
    clear_index()
    for f in UPLOAD_DIR.iterdir():
        f.unlink(missing_ok=True)

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    num_chunks = add_document_to_index(str(dest))
    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "chunks_indexed": num_chunks,
    }


@app.delete("/documents")
async def delete_all_documents():
    clear_index()
    for f in UPLOAD_DIR.iterdir():
        f.unlink(missing_ok=True)
    return {"message": "All documents cleared."}


@app.post("/voice")
async def voice_to_text(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    ext = Path(audio.filename).suffix.lstrip(".")

    import tempfile
    suffix = f".{ext or 'webm'}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio.filename, f.read()),
                model="whisper-large-v3",
            )
        return {"text": transcription.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import os as _os
        _os.unlink(tmp_path)


@app.get("/domains")
async def list_domains():
    from domain_config import DOMAIN_PROMPTS
    return {"domains": list(DOMAIN_PROMPTS.keys())}


@app.get("/health")
async def health():
    return {"status": "ok", "provider": "groq"}


# ── INTERVIEW ROUTES ──────────────────────────────────────────────────────────

@app.post("/interview/start")
async def start_interview(req: InterviewStart):
    questions = parse_resume_and_generate_questions(req.resume_text, req.job_role)
    return {"questions": questions}


@app.post("/interview/coding-questions")
async def get_coding_questions(req: CodingQuestions):
    questions = generate_coding_questions(req.job_role, req.language)
    return {"questions": questions}


@app.post("/interview/evaluate-answer")
async def evaluate_behavioral(req: AnswerEval):
    result = evaluate_answer(req.question, req.answer, req.job_role)
    return result


@app.post("/interview/evaluate-code")
async def evaluate_code_submission(req: CodeEval):
    result = evaluate_code(req.problem, req.code, req.language)
    return result


@app.post("/interview/report")
async def get_report(req: FinalReport):
    report = generate_final_report(
        req.job_role,
        req.behavioral_scores,
        req.coding_scores,
        req.candidate_name
    )
    return {"report": report}