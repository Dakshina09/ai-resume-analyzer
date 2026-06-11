import re
import fitz
import pandas as pd
import streamlit as st
import google.generativeai as genai

from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Career Intelligence System", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

SKILL_ALIASES = {
    "python": ["python", "py"],
    "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "excel": ["excel", "spreadsheet", "pivot table"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network"],
    "nlp": ["nlp", "natural language processing"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "keras"],
    "data visualization": ["tableau", "power bi", "matplotlib", "seaborn", "plotly"],
    "statistics": ["statistics", "probability", "a/b testing"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "react": ["react", "reactjs"],
    "node.js": ["node", "node.js", "express"],
    "git": ["git", "github"],
    "ci/cd": ["ci/cd", "github actions", "jenkins"],
    "system design": ["system design", "distributed systems"],
    "llm": ["llm", "large language model", "gpt", "gemini"],
    "rag": ["rag", "retrieval augmented generation", "vector database"]
}

ROLE_PROFILES = {
    "Data Analyst": ["sql", "excel", "python", "statistics", "data visualization"],
    "Data Scientist": ["python", "sql", "machine learning", "statistics", "data visualization"],
    "ML Engineer": ["python", "machine learning", "deep learning", "pytorch", "tensorflow", "docker", "fastapi"],
    "AI Engineer": ["python", "llm", "rag", "machine learning", "nlp", "fastapi"],
    "Backend Developer": ["python", "node.js", "sql", "fastapi", "django", "git", "system design"],
    "Full Stack Developer": ["react", "node.js", "sql", "git", "python"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "ci/cd", "git"],
    "MLOps Engineer": ["machine learning", "docker", "kubernetes", "aws", "ci/cd", "fastapi"]
}

def extract_pdf_text(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text() + "\n"
    return text

def extract_docx_text(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    if name.endswith(".docx"):
        return extract_docx_text(uploaded_file)

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    return ""

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_skills(text):
    text_l = text.lower()
    found = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias.lower()) + r"\b", text_l):
                found.add(skill)

    return sorted(found)

def semantic_similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0

    emb = model.encode([a, b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def section_score(text, keywords):
    text_l = text.lower()
    hits = sum(1 for k in keywords if k.lower() in text_l)
    return min(100, round((hits / max(len(keywords), 1)) * 100))

def calculate_ats_score(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = sorted(set(resume_skills) & set(jd_skills))
    missing = sorted(set(jd_skills) - set(resume_skills))

    skill_score = round(len(matched) / max(len(jd_skills), 1) * 100)
    semantic_score = round(semantic_similarity(resume_text, jd_text) * 100)

    impact_score = section_score(
        resume_text,
        ["increased", "reduced", "improved", "optimized", "%", "accuracy", "revenue"]
    )

    project_score = section_score(
        resume_text,
        ["project", "built", "developed", "deployed", "github", "api", "model"]
    )

    formatting_score = section_score(
        resume_text,
        ["education", "experience", "skills", "projects"]
    )

    ats = round(
        skill_score * 0.40 +
        semantic_score * 0.25 +
        impact_score * 0.15 +
        project_score * 0.10 +
        formatting_score * 0.10
    )

    return {
        "ats_score": ats,
        "skill_score": skill_score,
        "semantic_score": semantic_score,
        "impact_score": impact_score,
        "project_score": project_score,
        "formatting_score": formatting_score,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing
    }

def role_fit_scores(resume_text):
    resume_skills = set(extract_skills(resume_text))
    scores = {}

    for role, skills in ROLE_PROFILES.items():
        required = set(skills)
        scores[role] = round(len(resume_skills & required) / len(required) * 100)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

def ai_feedback(resume_text, jd_text, result, target_role):
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        return "GEMINI_API_KEY is not configured in Streamlit secrets."

    try:
        genai.configure(api_key=api_key)

        prompt = f"""
You are a senior AI recruiter and career coach.

Target role: {target_role}
ATS score: {result['ats_score']}/100
Matched skills: {result['matched_skills']}
Missing skills: {result['missing_skills']}

Resume:
{resume_text[:5000]}

Job description:
{jd_text[:3000]}

Write a premium report with:
1. Executive summary
2. Score explanation
3. Strong areas
4. Weak areas
5. Missing skills
6. Resume improvement suggestions
7. Project suggestions
8. 30-day roadmap
9. Final recruiter-style verdict

Be specific. Do not invent experience.
"""

        available_models = []

        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                available_models.append(model.name.replace("models/", ""))

        preferred_models = [
            "gemini-flash-latest",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro"
        ]

        model_names = preferred_models + available_models

        last_error = None

        for model_name in model_names:
            try:
                llm = genai.GenerativeModel(model_name)
                response = llm.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = e

        return f"AI feedback failed. Available models found: {available_models}. Last error: {last_error}"

    except Exception as e:
        return f"AI feedback failed: {e}"

st.title("AI Career Intelligence System")
st.caption("Resume analyzer, ATS scorer, role matcher, and AI career coach.")

with st.sidebar:
    target_role = st.selectbox("Target Role", list(ROLE_PROFILES.keys()), index=1)
    use_ai = st.toggle("Enable AI Career Coach", value=True)

resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
jd_text = st.text_area("Paste Job Description", height=260)

if st.button("Analyze Resume", type="primary"):
    if not resume_file:
        st.error("Please upload a resume.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analyzing resume..."):
            resume_text = clean_text(extract_text(resume_file))
            jd_text_clean = clean_text(jd_text)
            result = calculate_ats_score(resume_text, jd_text_clean)
            roles = role_fit_scores(resume_text)

        st.metric("Overall ATS Score", f"{result['ats_score']}/100")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Skills", result["skill_score"])
        c2.metric("Semantic", result["semantic_score"])
        c3.metric("Impact", result["impact_score"])
        c4.metric("Projects", result["project_score"])
        c5.metric("Structure", result["formatting_score"])

        col1, col2 = st.columns(2)

        col1.success("Matched Skills")
        col1.write(", ".join(result["matched_skills"]) or "No strong matches found.")

        col2.error("Missing Skills")
        col2.write(", ".join(result["missing_skills"]) or "No major missing skills detected.")

        st.subheader("Role Fit Scores")
        role_df = pd.DataFrame({
            "Role": list(roles.keys()),
            "Fit Score": list(roles.values())
        })
        st.dataframe(role_df, use_container_width=True)

        if use_ai:
            st.subheader("AI Career Coach Feedback")
            with st.spinner("Generating AI feedback..."):
                st.markdown(ai_feedback(resume_text, jd_text_clean, result, target_role))