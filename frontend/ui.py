import streamlit as st
import requests

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="centered"
)

# Header
st.title("AI Resume Screening System")
st.caption("Upload a resume and compare it with a job description")

st.markdown("---")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# Job description input
job_description = st.text_area(
    "Paste Job Description",
    height=180,
    placeholder="Example: Looking for Python, SQL, Machine Learning, FastAPI..."
)

# Analyze button
if st.button("Analyze Resume", use_container_width=True):

    if uploaded_file and job_description:

        with st.spinner("Analyzing resume..."):

            response = requests.post(
                "http://127.0.0.1:8000/match",
                files={"resume": uploaded_file},
                data={"job_description": job_description}
            )

        if response.status_code == 200:

            result = response.json()

            st.markdown("---")
            st.subheader("Analysis Result")

            score = result["score"]

            st.metric("Match Score", f"{score}%")

            if result["label"] == "Strong match for this role":
                st.success(result["label"])
            elif result["label"] == "Moderate match":
                st.warning(result["label"])
            else:
                st.error(result["label"])

            st.markdown("### Recruiter Feedback")
            st.info(result["feedback"])

            st.markdown("### Missing Skills")

            if result["missing_skills"]:
                cols = st.columns(3)
                for i, skill in enumerate(result["missing_skills"]):
                    cols[i % 3].error(skill)
            else:
                st.success("No major missing skills detected")

    else:
        st.warning("Upload resume and enter job description")