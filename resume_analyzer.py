import fitz  # PyMuPDF
import streamlit as st
import spacy
from spacy.cli import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# ✅ Streamlit-safe way to ensure spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

skill_keywords = ['python', 'java', 'sql', 'machine learning', 'deep learning', 'data analysis',
                  'tensorflow', 'communication', 'leadership', 'problem solving', 'teamwork']

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

def extract_skills(text):
    return list(set([skill for skill in skill_keywords if skill in text]))

def match_score(resume_text, job_desc):
    corpus = [resume_text, job_desc.lower()]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(corpus)
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

def generate_report(name, email, resume_skills, job_skills, score):
    missing = list(set(job_skills) - set(resume_skills))
    return f"""
    Resume Analysis Report
    -----------------------
    Candidate Name: {name}
    Email: {email}

    Match Score: {score}%

    Extracted Skills from Resume:
    {', '.join(resume_skills) if resume_skills else 'None'}

    Expected Skills from Job Description:
    {', '.join(job_skills) if job_skills else 'None'}

    Missing Skills:
    {', '.join(missing) if missing else 'None'}
    """

def create_download_button(content):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="resume_report.txt">📄 Download Report</a>'

st.title("🤖 AI Resume Analyzer")

name = st.text_input("Candidate Name")
email = st.text_input("Email Address")
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume_file and job_desc:
        resume_text = extract_text_from_pdf(resume_file)
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_desc)
        score = match_score(resume_text, job_desc)
        report = generate_report(name, email, resume_skills, job_skills, score)

        st.success(f"Match Score: {score}%")
        st.markdown("### 📄 Report")
        st.text(report)
        st.markdown(create_download_button(report), unsafe_allow_html=True)
    else:
        st.error("Please upload a resume and paste the job description.")
