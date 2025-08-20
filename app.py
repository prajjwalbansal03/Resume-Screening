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

# -------------------------------
# Hardcoded Interview Question Generator
# -------------------------------
def generate_interview_questions(role: str, matched_skills: list) -> list:
    """
    Instead of using an API, generate 1-5 hardcoded hardcore questions
    based on skills for demonstration purposes.
    """
    questions_bank = {
        "python": [
            "Explain the difference between deep copy and shallow copy in Python.",
            "How do you manage memory in Python for large datasets?"
        ],
        "sql": [
            "Write a SQL query to find the second highest salary in a table.",
            "Explain the difference between INNER JOIN and OUTER JOIN."
        ],
        "javascript": [
            "Explain closures in JavaScript and give an example.",
            "What is event delegation and why is it useful?"
        ],
        "react": [
            "How does the virtual DOM work in React?",
            "Explain React hooks and their usage."
        ],
        "aws": [
            "Explain the difference between S3 and EBS in AWS.",
            "How would you secure an AWS Lambda function?"
        ],
        "c++": [
        "Explain the difference between pointers and references in C++.",
        "What is RAII and why is it important?"
        ],
        "java": [
            "Explain the difference between JDK, JRE, and JVM.",
            "What is the difference between an abstract class and an interface in Java?"
        ]
    }

    generated = []
    for skill in matched_skills:
        skill = skill.lower()
        if skill in questions_bank:
            generated.append(questions_bank[skill][0])  # just pick 1 question per skill

    if not generated:
        generated = [f"Prepare a practical question related to {', '.join(matched_skills)}."]

    return generated[:5]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Resume Screening Dashboard", page_icon="📄", layout="wide")
st.title("📄 Resume Screening Dashboard")
st.caption("Upload resumes once, define multiple job roles & skills, and rank candidates per role.")

with st.sidebar:
    st.header("⚙️ Settings")
    use_fuzzy = st.toggle("Enable fuzzy matching", value=True)
    fuzzy_threshold = st.slider("Fuzzy threshold", 70, 100, 85)

st.markdown("### 1) Define Job Roles & Required Skills")

if "job_roles" not in st.session_state:
    st.session_state.job_roles = []

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

            # Generate interview questions per candidate (hardcoded)
            df["Interview Questions"] = df.apply(
                lambda row: "\n".join(generate_interview_questions(role, row["Matched Skills"].split(", "))),
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
