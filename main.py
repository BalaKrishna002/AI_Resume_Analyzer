from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from analyzer import analyze_resume
from utils import extract_text_from_pdf

app = FastAPI(title="AI Resume Analyzer")

templates = Jinja2Templates(directory="templates")


# ---------------- Home Route ---------------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- Analyze Route ---------------- #

@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    # Validate PDF
    if resume.content_type != "application/pdf":
        return {"error": "Only PDF files are supported"}

    # Extract text
    resume_text = extract_text_from_pdf(resume.file)

    if not resume_text or not resume_text.strip():
        return {"error": "Could not extract text from PDF"}

    # Run Analyzer (now returns dict)
    result = analyze_resume(resume_text, job_description)

    # Directly return dictionary
    return result