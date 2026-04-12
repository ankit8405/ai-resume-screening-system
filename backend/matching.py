import pdfplumber
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

skills_df = pd.read_csv("../data/skills.csv")
skills_list = skills_df["skill"].tolist()

degree_df = pd.read_csv("../data/education_degree.csv")
degree_list = degree_df["degree"].tolist()

domain_df = pd.read_csv("../data/education_domain.csv")
domain_list = domain_df["domain"].tolist()

section_labels = {
    "skills": "technical skills programming tools machine learning frameworks",
    "education": "degree bachelor master academic qualification university education",
    "projects": "projects built implemented developed portfolio work",
    "experience": "experience internship work years professional background"
}

section_embeddings = {
    section: model.encode(text, convert_to_tensor=True)
    for section, text in section_labels.items()
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_text_from_pdf(file):
    text = ""

    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    return text

def split_resume_sections(text):

    sections = {
        "skills": "",
        "education": "",
        "projects": "",
        "experience": ""
    }

    current_section = None

    lines = text.split("\n")

    for line in lines:

        if len(line.strip()) < 3:
            continue

        if len(line.split()) <= 4:
            cleaned_line = clean_text(line)
            line_embedding = model.encode(cleaned_line, convert_to_tensor=True)

            scores = {}

            for label, emb in section_embeddings.items():
                score = util.cos_sim(line_embedding, emb).item()
                scores[label] = score

            best_section = max(scores, key=scores.get)

            # threshold avoids accidental matches
            if len(line.split()) <= 4 and scores[best_section] > 0.45:
                current_section = best_section
                continue

        if current_section:
            sections[current_section] += line + " "

    return sections

def detect_jd_weights(job_description):

    jd_embedding = model.encode(job_description, convert_to_tensor=True)

    scores = {}

    for section, emb in section_embeddings.items():
        score = util.cos_sim(jd_embedding, emb).item()
        scores[section] = max(score, 0)

    total = sum(scores.values())

    if total == 0:
        return {
            "skills": 0.25,
            "education": 0.25,
            "projects": 0.25,
            "experience": 0.25
        }

    return {
        section: score / total
        for section, score in scores.items()
    }

def detect_jd_type(job_description):

    jd_skills = extract_skills(job_description)
    words = job_description.split()

    skill_ratio = len(jd_skills) / max(len(words), 1)

    if len(words) <= 15 and len(jd_skills) >= 5:
        return "skill_heavy"

    elif len(jd_skills) >= 4 and skill_ratio > 0.10:
        return "mixed"

    return "descriptive"

def extract_education_degree(text):

    text = clean_text(text)

    found = []

    for degree in degree_list:
        normalized_degree = clean_text(degree)

        if re.search(rf'\b{re.escape(normalized_degree)}\b', text):
            found.append(degree)

    return found

def extract_education_domain(text):
    text = clean_text(text)

    found = []

    for domain in domain_list:
        normalized_domain = clean_text(domain)

        if re.search(rf'\b{re.escape(normalized_domain)}\b', text):
            found.append(domain)

    return found

def extract_skills(text):
    text = clean_text(text)

    found = []

    for skill in skills_list:
        normalized_skill = clean_text(skill)

        if re.search(rf'\b{re.escape(normalized_skill)}\b', text):
            found.append(skill)

    return found

def normalize_aliases(text):

    aliases = {
        "cse": "computer science and engineering",
        "cs": "computer science",
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "ece": "electronics and communication engineering",
        "eee": "electrical and electronics engineering",
        "it": "information technology"
    }

    text = clean_text(text)

    for short, full in aliases.items():
        text = re.sub(rf'\b{short}\b', full, text)

    return text

def calculate_similarity(resume_text, job_description):

    sections = split_resume_sections(resume_text)

    jd_embedding = model.encode(clean_text(job_description), convert_to_tensor=True)

    weights = detect_jd_weights(job_description)

    semantic_score = 0
    

    for section, content in sections.items():

        if content.strip():

            emb = model.encode(clean_text(content), convert_to_tensor=True)

            section_similarity = util.cos_sim(jd_embedding, emb).item()

            semantic_score += section_similarity * weights[section]
    
    semantic_score = min(semantic_score, 0.90)

    resume_degree = extract_education_degree(resume_text)
    jd_degree = extract_education_degree(job_description)

    resume_domain = extract_education_domain(normalize_aliases(resume_text))
    jd_domain = extract_education_domain(normalize_aliases(job_description))

    education_score = 0
    degree_match = 0
    domain_match = 0

    if jd_degree:
        degree_match = len(set(resume_degree) & set(jd_degree))
        if degree_match:
            education_score += 0.05
        else:
            education_score -= 0.05

    if jd_domain:
        domain_match = len(set(resume_domain) & set(jd_domain))
        if domain_match  > 0:
            education_score += 0.10
        else:
            education_score -= 0.05

    if jd_degree or jd_domain:
        resume_edu_text = " ".join(resume_degree + resume_domain)
        jd_edu_text = " ".join(jd_degree + jd_domain)

        if resume_edu_text and jd_edu_text:
            edu_semantic = util.cos_sim(
                model.encode(resume_edu_text, convert_to_tensor=True),
                model.encode(jd_edu_text, convert_to_tensor=True)
            ).item()

            education_score += edu_semantic * 0.05

    education_score = max(min(education_score, 0.15), -0.15)

    project_score = 0

    if sections["projects"].strip():

        project_embedding = model.encode(
            clean_text(sections["projects"]),
            convert_to_tensor=True
        )

        project_similarity = util.cos_sim(jd_embedding, project_embedding).item()

        project_score = min(project_similarity, 0.85)

    role_words = [
        "intern", "developer", "engineer", "analyst", "scientist", "manager", "consultant", "specialist", "coordinator", "administrator", "architect", "designer", "programmer", "tester", "lead", "head", "director"
    ]

    project_words = [
        "project", "projects", "built", "developed", "implemented", "created", "designed", "launched", "deployed", "engineered", "research", "thesis", "capstone", "contributed", "collaborated", "led", "managed", "coordinated", 
        "organized", "executed", "delivered", "achieved", "accomplished", "innovated", "optimized", "improved", "automated", "streamlined", "enhanced", "scaled", "published"
    ]

    contains_role = any(
        word in clean_text(job_description)
        for word in role_words
    )

    contains_project = any(
        word in clean_text(job_description)
        for word in project_words
    )

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)
    missing_skills = []

    if jd_skills:
        matched = len(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))
        missing = len(set(jd_skills) - set(resume_skills))
        skill_score = max((matched - 0.25 * missing) / len(jd_skills), 0)
    else:
        skill_score = 0

    if len(job_description.split()) <= 10:

        semantic_score = util.cos_sim(
            model.encode(clean_text(resume_text), convert_to_tensor=True),
            model.encode(clean_text(job_description), convert_to_tensor=True)
        ).item()

        if contains_role:

            final_score = (
                0.35 * semantic_score +
                0.50 * skill_score +
                0.15 * education_score
            ) * 100

        elif contains_project:

            final_score = (
                0.60 * project_score +
                0.25 * semantic_score +
                0.15 * skill_score
            ) * 100

        elif jd_skills:

            final_score = (
                0.70 * skill_score +
                0.20 * semantic_score +
                0.10 * project_score
            ) * 100

        elif jd_degree or jd_domain:

            normalized_education = education_score / 0.15 if education_score > 0 else 0

            final_score = (
                0.85 * normalized_education +
                0.15 * semantic_score
            ) * 100

        else:

            final_score = semantic_score * 100

    else:

        if jd_skills:
            matched = len(set(resume_skills) & set(jd_skills))
            missing = len(set(jd_skills) - set(resume_skills))
            skill_score = max((matched - 0.3 * missing) / len(jd_skills), 0)
        else:
            skill_score = 0

        jd_type = detect_jd_type(job_description)

        if jd_type == "skill_heavy":
            final_score = (
                0.75 * skill_score +
                0.15 * semantic_score +
                0.05 * education_score +
                0.05 * project_score
            ) * 100

        elif jd_type == "mixed":
            final_score = (
                0.50 * skill_score +
                0.25 * semantic_score +
                0.10 * education_score +
                0.15 * project_score
            ) * 100

        else:
            final_score = (
                0.35 * semantic_score +
                0.35 * skill_score +
                0.12 * education_score +
                0.18 * project_score
            ) * 100

    final_score = round(min(max(final_score, 0), 100), 2)

    if final_score >= 80:
        label = "Strong match for this role"
    elif final_score >= 65:
        label = "Moderate match"
    elif final_score >= 45:
        label = "Weak to moderate match"
    else:
        label = "Low match"

    missing_skills = [
        skill for skill in missing_skills
        if len(skill) > 2
    ][:5]

    feedback = []

    if skill_score >= 0.7:
        feedback.append("Strong technical skill alignment.")
    elif skill_score >= 0.4:
        feedback.append("Good skill base, but some required skills are missing.")
    else:
        feedback.append("Limited skill overlap with the job description.")

    if education_score > 0:
        feedback.append("Educational background aligns well.")

    if project_score > 0.08:
        feedback.append("Projects strongly support the target role.")

    if semantic_score >= 0.7:
        feedback.append("Resume content closely matches role intent.")

    feedback = " ".join(feedback)

    return {
        "score": final_score,
        "label": label,
        "missing_skills": missing_skills,
        "feedback": feedback
    }