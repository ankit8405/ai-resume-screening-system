# ATS Resume Expert

A Streamlit web app that uses **Google Gemini (vision)** to evaluate a resume against a job
description — like an Applicant Tracking System (ATS) scanner.

## What it does

1. Paste a **job description**.
2. Upload a **resume as a PDF**.
3. Click one of two analysis buttons:
   - **Tell Me About the Resume** — a technical recruiter's evaluation: how well the candidate
     aligns with the role, plus their strengths and weaknesses.
   - **Percentage match** — an ATS-style score: percentage match, missing keywords, and final thoughts.

The resume PDF's pages are rendered to images and sent to Gemini together with the job description — up to **2 pages**. Longer resumes are rejected with a message asking for a shorter version.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini API |
| PDF → image | `pdf2image` (requires the **poppler** system binary) |

Model in use: **`gemini-3.6-flash`** (set in `app.py`).

## Setup

1. **Install poppler** (required by `pdf2image` to read PDFs):
   - macOS: `brew install poppler`
   - Ubuntu/Debian: `sudo apt-get install poppler-utils`
   - Windows: install poppler and pass its path via pdf2image's `poppler_path`

2. **Install the Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Google Gemini API key** to a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your-key-here
   ```
   Get a key at https://aistudio.google.com/apikey

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Project structure

```
.
├── app.py            # the whole app (Streamlit UI + Gemini calls)
├── requirements.txt  # Python dependencies
├── .env              # GOOGLE_API_KEY (git-ignored, not committed)
└── README.md
```

## Configuration

- **`GOOGLE_API_KEY`** — read from `.env` (required).
- **`MODEL`** — in `app.py`, defaults to `gemini-3.6-flash`. If Google retires it you'll get a 404;
  update this constant (e.g. to `gemini-flash-latest`).
- **`MAX_PAGES`** — in `app.py`, the maximum number of resume pages analyzed (default `2`).

## Notes & limitations

- Up to **2 pages** of the resume are analyzed. Longer PDFs are rejected with a message asking for a resume of at most 2 pages.
- Gemini model names change over time — a 404 means the model was retired; update `MODEL`.
- Errors (missing key, unreadable PDF, API/quota failures) appear as friendly messages in the UI instead of raw tracebacks.
