from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from matching import extract_text_from_pdf, calculate_similarity

app = FastAPI()


@app.post("/match")
async def match_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    resume_text = extract_text_from_pdf(resume)

    result = calculate_similarity(resume_text, job_description)

    return JSONResponse(result)