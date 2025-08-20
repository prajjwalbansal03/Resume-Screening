import io
import re
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
from pymongo import MongoClient

# -------------------------------
# MongoDB Connection
# -------------------------------
MONGO_URI = st.secrets["MONGO"]["MONGO_URI"]

@st.cache_resource
def get_mongo_client():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Will raise if cannot connect
    client.server_info()
    return client

try:
    client = get_mongo_client()
    db = client["resume_screening"]
    job_roles_collection = db["job_roles"]   # one document per user: { user_id, roles: {...} }
except Exception as e:
    st.error(f"Could not connect to MongoDB: {e}")
    st.stop()

# -------------------------------
# Optional libs
# -------------------------------
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
# Helpers – data layer
# -------------------------------
def save_job_roles_to_mongo(user_id: str, job_roles_data: dict):
    """Upsert a single document for this user."""
    job_roles_collection.update_one(
        {"user_id": user_id},
        {"$set": {"roles": job_roles_data}},
        upsert=True
    )

def load_job_roles_from_mongo(user_id: str) -> Dict[str, Dict[str, float]]:
    """Return {role: {skill: weight}} or {} if none."""
    doc = job_roles_collection.find_one({"user_id": user_id})
    if doc and "roles" in doc and isinstance(doc["roles"], dict):
        return doc["roles"]
    return {}

def delete_all_roles_for_user(user_id: str):
    job_roles_collection.update_one(
        {"user_id": user_id},
        {"$set": {"roles": {}}},
        upsert=True
    )

# -------------------------------
# Helpers – scoring/parsing
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
    """Parse 'python:1.0, sql:0.8, aws' -> {'python':1.0,'sql':0.8,'aws':1.0}"""
    if not skills_raw or not skills_raw.strip():
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

def generate_interview_questions(role: str, matched_skills: list) -> list:
    bank = {
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
    for s in matched_skills:
        s = s.lower().strip()
        if s in bank:
            generated.append(bank[s][0])
    if not generated:
        generated = [f"Prepare a practical question related to {', '.join(matched_skills)}."]
    return generated[:5]

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Resume Screening Dashboard", page_icon="📄", layout="wide")
st.title("📄 Resume Screening Dashboard")
st.caption("Upload resumes once, define multiple job roles & skills, and rank candidates per role.")

# -------------------------------
# Login / Logout with clean session handoff
# -------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

def reset_user_state():
    for k in ("job_roles", "roles_skills"):
        if k in st.session_state:
            del st.session_state[k]

if not st.session_state.user_id:
    st.markdown("### 🔐 Login")
    username = st.text_input("Enter your username or email")
    if st.button("Login"):
        if username.strip():
            # new login: clear any previous user data to avoid bleed
            reset_user_state()
            st.session_state.user_id = username.strip()
            st.rerun()
        else:
            st.warning("Please enter a valid username to continue.")
    st.stop()

user_id = st.session_state.user_id
col_a, col_b = st.columns([4,1])
with col_a:
    st.success(f"✅ Logged in as **{user_id}**")
with col_b:
    if st.button("🚪 Logout"):
        reset_user_state()
        st.session_state.user_id = ""
        st.rerun()

# -------------------------------
# Sidebar Settings
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_fuzzy = st.toggle("Enable fuzzy matching", value=True)
    fuzzy_threshold = st.slider("Fuzzy threshold", 70, 100, 85)

# -------------------------------
# Load roles for THIS user only
# -------------------------------
roles_from_db = load_job_roles_from_mongo(user_id)

# Initialize session state for this user (fresh OR from DB)
if "roles_skills" not in st.session_state:
    st.session_state.roles_skills = roles_from_db.copy()  # local working copy
if "job_roles" not in st.session_state:
    st.session_state.job_roles = list(st.session_state.roles_skills.keys())

# Friendly banner
if roles_from_db:
    st.info(f"Loaded {len(roles_from_db)} role(s) from your saved profile.")
else:
    st.warning("New user profile detected. Start by adding job roles and skills, then click **Save All Changes**.")

# -------------------------------
# Define roles & skills (no auto-save)
# -------------------------------
st.markdown("### 1) Define Job Roles & Required Skills")

with st.form("roles_form", clear_on_submit=False):
    c1, c2 = st.columns([3, 1])
    with c1:
        new_role = st.text_input("Add a new job role")
    with c2:
        add_clicked = st.form_submit_button("➕ Add Role")
    if add_clicked and new_role:
        if new_role not in st.session_state.job_roles:
            st.session_state.job_roles.append(new_role)
            st.session_state.roles_skills.setdefault(new_role, {})

    # Editable skills per role
    remove_marks = []
    for role in st.session_state.job_roles:
        row1, row2 = st.columns([4,1])
        with row1:
            skills_raw = st.text_input(
                f"Skills for {role} (comma-separated, weights optional)",
                value=", ".join([f"{k}:{v}" for k, v in st.session_state.roles_skills.get(role, {}).items()]),
                key=f"skills_{role}"
            )
            st.session_state.roles_skills[role] = parse_weighted_skills(skills_raw)
        with row2:
            if st.form_submit_button(f"🗑️ Remove {role}", use_container_width=True):
                remove_marks.append(role)

    # remove after loop to avoid widget conflicts
    for r in remove_marks:
        st.session_state.job_roles = [x for x in st.session_state.job_roles if x != r]
        st.session_state.roles_skills.pop(r, None)

    save_clicked = st.form_submit_button("💾 Save All Changes", use_container_width=True)

if save_clicked:
    save_job_roles_to_mongo(user_id, st.session_state.roles_skills)
    st.success("Saved your roles & skills.")
    st.rerun()

# Quick action buttons
colx, coly = st.columns([1,2])
with colx:
    if st.button("🧹 Start From Scratch (Clear All)"):
        st.session_state.roles_skills = {}
        st.session_state.job_roles = []
        delete_all_roles_for_user(user_id)
        st.rerun()

# -------------------------------
# Upload resumes
# -------------------------------
st.markdown("### 2) Upload Resumes (.pdf / .docx / .txt)")
files = st.file_uploader(
    "Upload multiple resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

st.markdown("---")
process = st.button("🔎 Process & Rank Candidates", type="primary", use_container_width=True)

# Use current working copy; if user never saved, they still can test locally
current_roles = {r: st.session_state.roles_skills.get(r, {}) for r in st.session_state.job_roles}

if process:
    if not files:
        st.error("Please upload at least one resume file.")
    elif not any(current_roles.values()):
        st.error("Please add at least one role with skills.")
    else:
        all_results = {}
        for role, skills in current_roles.items():
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
                lambda row: "\n".join(
                    generate_interview_questions(role, [s for s in row["Matched Skills"].split(", ") if s])
                ),
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
