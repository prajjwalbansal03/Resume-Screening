
import io
import re
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


from google import genai
import os
client = genai.Client(api_key=st.secrets["GEMINI"]["API_KEY"])




st.set_page_config(page_title="Resume Screening Dashboard", page_icon="📄", layout="wide")

# -------------------------------
# Helpers
# -------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text_parts = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(text_parts)
    except Exception as e:
        st.warning(f"Could not extract text from PDF: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        return ""
    with io.BytesIO(file_bytes) as fh:
        document = docx.Document(fh)
        return "\n".join([p.text for p in document.paragraphs])

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def parse_weighted_skills(skills_raw: str) -> Dict[str, float]:
    if not skills_raw.strip():
        return {}
    parts = [p.strip() for p in skills_raw.split(",") if p.strip()]
    out = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            try:
                out[k.strip().lower()] = float(v.strip())
            except Exception:
                out[k.strip().lower()] = 1.0
        else:
            out[p.strip().lower()] = 1.0
    return out

def match_skill_in_text(skill: str, text: str, use_fuzzy: bool, fuzzy_threshold: int) -> bool:
    if skill in text:
        return True
    if use_fuzzy and fuzz is not None:
        score = fuzz.partial_ratio(skill, text)
        return score >= fuzzy_threshold
    return False

def score_resume(resume_text: str, required_skills: Dict[str, float], use_fuzzy=True, fuzzy_threshold=85) -> Tuple[float, List[str]]:
    text = clean_text(resume_text)
    matched = []
    total = 0.0
    for skill, w in required_skills.items():
        if match_skill_in_text(skill, text, use_fuzzy, fuzzy_threshold):
            total += w
            matched.append(skill)
    return total, matched

@st.cache_data(show_spinner=False)
def extract_resume_text(file) -> str:
    content = file.read()
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif name.endswith(".txt"):
        return extract_text_from_txt(content)
    else:
        return ""

def generate_interview_questions_gemini(role: str, matched_skills: list) -> list:
    if not matched_skills:
        return []

    prompt = f"""
You are an HR expert. Generate 3-5 interview questions for a candidate applying for '{role}'.
The candidate has the following skills: {', '.join(matched_skills)}.
Make the questions practical and relevant to the skills.
Start sentence directly with the question.
"""

    try:
        response = client.generate_text(
            model="gemini-2.5-turbo",
            prompt=prompt,
            max_output_tokens=300
        )
        questions_text = response.text
        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        return questions
    except Exception as e:
        return [f"Error generating questions: {e}"]


# -------------------------------
# UI
# -------------------------------
st.title("📄 Resume Screening Dashboard")
st.caption("Upload resumes once, define multiple job roles & skills, and rank candidates per role.")

with st.sidebar:
    st.header("⚙️ Settings")
    use_fuzzy = st.toggle("Enable fuzzy matching", value=True)
    fuzzy_threshold = st.slider("Fuzzy threshold", 70, 100, 85)

st.markdown("### 1) Define Job Roles & Required Skills")


if "job_roles" not in st.session_state:
    st.session_state.job_roles = []  # start empty

col1, col2 = st.columns([3, 1])
with col1:
    new_role = st.text_input("Add a new job role")
with col2:
    if st.button("➕ Add Role") and new_role:
        st.session_state.job_roles.append(new_role)

roles_skills = {}
for role in st.session_state.job_roles:
    skills_raw = st.text_input(f"Skills for {role} (comma-separated, weights optional)", key=role)
    roles_skills[role] = parse_weighted_skills(skills_raw)

st.markdown("### 2) Upload Resumes (.pdf / .docx / .txt)")
files = st.file_uploader(
    "Upload multiple resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

st.markdown("---")
process = st.button("🔎 Process & Rank Candidates", type="primary", use_container_width=True)

if process:
    if not files:
        st.error("Please upload at least one resume file.")
    else:
        all_results = {}
        for role, skills in roles_skills.items():
            if not skills:
                continue
            rows = []
            for f in files:
                text = extract_resume_text(f)
                score, matched = score_resume(text, skills, use_fuzzy, fuzzy_threshold)
                rows.append({
                    "Candidate": f.name,
                    "Total Score": score,
                    "Matched Skills": ", ".join(matched)
                })
            df = pd.DataFrame(rows).sort_values(by="Total Score", ascending=False)

            # Generate interview questions per candidate
            df["Interview Questions"] = df.apply(
                lambda row: "\n".join(generate_interview_questions_gemini(role, row["Matched Skills"].split(", "))),
                axis=1
            )

            all_results[role] = df

        if all_results:
            st.markdown("### 3) Results by Job Role")
            tabs = st.tabs(list(all_results.keys()))
            for (role, df), tab in zip(all_results.items(), tabs):
                with tab:
                    st.subheader(f"Role: {role}")
                    st.dataframe(df, use_container_width=True)

                   
                    for idx, row in df.iterrows():
                        with st.expander(f"Interview Questions for {row['Candidate']}"):
                            st.text(row["Interview Questions"])

                    st.download_button(
                        label=f"⬇️ Download {role} Results (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=f"results_{role.replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
