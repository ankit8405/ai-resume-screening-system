import io
import os

import pdf2image
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pdf2image.exceptions import (
    PDFPageCountError,
    PDFSyntaxError,
    PopplerNotInstalledError,
)

load_dotenv()

MODEL = "gemini-3.6-flash"
MAX_PAGES = 2

_client: genai.Client | None = None


class AppError(Exception):
    pass


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise AppError("No API key found. Add GOOGLE_API_KEY=... to your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def _pdf_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except PopplerNotInstalledError as err:
        raise AppError(
            "Can't read PDFs — poppler is not installed. On macOS run: brew install poppler"
        ) from err
    except (PDFPageCountError, PDFSyntaxError) as err:
        raise AppError("Can't read the PDF — it may be corrupted or password-protected.") from err
    except Exception as err:
        raise AppError(f"Can't read the PDF: {err}") from err


def _friendly_api_error(err: genai_errors.APIError) -> str:
    code = getattr(err, "code", None)
    if code == 404:
        return "Model unavailable — the configured model has been retired. Update MODEL in app.py."
    if code == 429:
        return "Rate limit or quota exceeded. Please wait a moment and try again."
    if code in (401, 403):
        return "Authentication failed — check that GOOGLE_API_KEY is valid."
    if code and code >= 500:
        return "Gemini had a temporary server error. Please try again."
    return f"Gemini request failed: {err}"


def get_gemini_response(
    instruction: str, resume_pages: list[types.Part], job_description: str
) -> str:
    try:
        response = get_client().models.generate_content(
            model=MODEL,
            contents=[instruction, *resume_pages, job_description],
        )
    except genai_errors.APIError as err:
        raise AppError(_friendly_api_error(err)) from err

    text = response.text
    if text:
        return text
    if response.candidates:
        return f"(No response text — model stopped: {response.candidates[0].finish_reason})"
    return "(No response — the prompt was blocked by safety filters.)"


def input_pdf_setup(uploaded_file) -> list[types.Part]:
    if uploaded_file is None:
        raise AppError("No file uploaded.")

    pdf_bytes = uploaded_file.read()
    info = _pdf_call(pdf2image.pdfinfo_from_bytes, pdf_bytes)
    page_count = int(info.get("Pages", 1))

    if page_count > MAX_PAGES:
        raise AppError(
            f"This resume is {page_count} pages long. Please upload a shorter resume "
            f"of at most {MAX_PAGES} pages."
        )

    images = _pdf_call(pdf2image.convert_from_bytes, pdf_bytes, last_page=MAX_PAGES)

    pages = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        pages.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"))
    return pages


INPUT_PROMPT1 = """
 You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description.
  Please share your professional evaluation on whether the candidate's profile aligns with the role.
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

INPUT_PROMPT3 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality,
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""


def main() -> None:
    st.set_page_config(page_title="ATS Resume Expert")
    st.header("ATS Tracking System")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is not set. Add it to your .env file and reload the page.")
        st.stop()

    job_description = st.text_area("Job Description: ", key="input")
    uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

    if uploaded_file is not None:
        st.write("PDF Uploaded Successfully")

    submit1 = st.button("Tell Me About the Resume")
    submit3 = st.button("Percentage match")

    instruction = INPUT_PROMPT1 if submit1 else INPUT_PROMPT3 if submit3 else None
    if instruction is None:
        return

    if uploaded_file is None:
        st.error("Please upload a resume (PDF).")
        return
    if not job_description.strip():
        st.warning("Please enter a job description.")
        return

    try:
        with st.spinner("Analyzing resume..."):
            resume_pages = input_pdf_setup(uploaded_file)
            response = get_gemini_response(instruction, resume_pages, job_description)
    except AppError as err:
        st.error(str(err))
        return

    st.subheader("The Response is")
    st.write(response)


if __name__ == "__main__":
    main()
