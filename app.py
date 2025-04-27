import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader
import pandas as pd
import altair as alt
import base64
import re

# Map of class labels
class_labels = [
    "Software Engineer", "Data Scientist", "AI Researcher", "Frontend Developer",
    "Backend Developer", "Fullstack Developer", "Blockchain Developer",
    "Game Developer", "DevOps Engineer", "Cloud Engineer", "Mobile App Developer",
    "Embedded Systems Engineer", "IoT Specialist", "Hardware Engineer",
    "Network Engineer", "Database Administrator", "Cybersecurity Specialist",
    "Business Analyst", "Product Manager", "Project Manager", "Research Scientist",
    "Quality Assurance Engineer", "Technical Writer", "UI/UX Designer", "IT Support"
]

# Load model and tokenizer
@st.cache_resource
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained(".", local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(".", local_files_only=True)
    return model, tokenizer

# Predict job role
def predict_job_role(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    return probs

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Smart resume parsing
def extract_skills(text):
    skills_keywords = [
        "Python", "Java", "C++", "JavaScript", "SQL", "HTML", "CSS", "AWS", "Docker", 
        "Kubernetes", "Machine Learning", "Deep Learning", "DevOps", "Git", "CI/CD", "Terraform", "Linux"
    ]
    found_skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return found_skills

def extract_education(text):
    education_patterns = [
        r"\b(Bachelor|Master|PhD)\b[\w\s]*\b(Science|Engineering|Mathematics|Technology|Computer)\b",
        r"\bDegree\b[\w\s]*\b(Computer Science|Engineering|Information Technology|Software)\b"
    ]
    education = []
    for pattern in education_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        education.extend(matches)
    return education

def extract_experience(text):
    experience_keywords = [
        "Experience", "Work Experience", "Professional Experience", "Employment History"
    ]
    experience = []
    for keyword in experience_keywords:
        start_idx = text.lower().find(keyword.lower())
        if start_idx != -1:
            experience_text = text[start_idx:]
            experience.append(experience_text[:300])
    return experience

# Create download link
def create_download_link(text, filename="prediction.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Prediction Results</a>'
    return href

# Streamlit App
st.set_page_config(page_title="Resume Job Role Predictor", page_icon="📝")
st.title("📝 Resume Job Role Predictor")

st.write("Upload your resume (PDF or TXT) and see what job role suits you best!")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode('utf-8')

    st.subheader("📄 Extracted Resume Text")

    # Scrollable box for full resume text
    with st.expander("🔍 Click here to view the full resume text"):
        st.write(text)

    # Smart resume parsing
    st.subheader("🔑 Extracted Information")

    # Extract data
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)

    # Arrange into columns for better UI
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🛠️ Skills")
        if skills:
            with st.container():
                for skill in skills:
                    st.markdown(f"- {skill}")
        else:
            st.info("No skills found.")

    with col2:
        st.markdown("### 🎓 Education")
        if education:
            with st.container():
                for degree in education:
                    st.markdown(f"- {degree}")
        else:
            st.info("No education information found.")

    with col3:
        st.markdown("### 💼 Experience")
        if experience:
            with st.container():
                for exp in experience:
                    st.markdown(f"- {exp[:100]}...")  # Just show first 100 characters
        else:
            st.info("No work experience found.")

    st.divider()

    if st.button("🔍 Predict Job Role"):
        with st.spinner('Analyzing your resume...'):
            model, tokenizer = load_bert_model()
            probs = predict_job_role(text, model, tokenizer)

            top_idx = torch.argmax(torch.tensor(probs)).item()
            predicted_role = class_labels[top_idx]

            # Sort top 5
            sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            top5 = sorted_probs[:5]

            st.success(f"🔍 **Predicted Job Role: {predicted_role}**")

            st.subheader("🔢 Prediction Probabilities (Top 5)")
            for idx, score in top5:
                st.write(f"**{class_labels[idx]}:** {score:.4f}")

            # Bar chart
            labels = [class_labels[idx] for idx, _ in top5]
            scores = [score for _, score in top5]
            chart_data = pd.DataFrame({
                'Job Role': labels,
                'Probability': scores
            })

            st.subheader("📈 Top 5 Predicted Roles (Chart View)")
            bar_chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Job Role', sort='-y'),
                y='Probability',
                color=alt.Color('Job Role', legend=None)
            ).properties(width=600)
            st.altair_chart(bar_chart)

            # Download link
            result_text = f"Predicted Job Role: {predicted_role}\n\nTop 5 Predictions:\n"
            for idx, score in top5:
                result_text += f"{class_labels[idx]}: {score:.4f}\n"

            # Add extracted information to download link
            result_text += "\n**Extracted Information:**\n"
            result_text += f"Skills: {', '.join(skills) if skills else 'None found'}\n"
            result_text += f"Education: {', '.join(education) if education else 'None found'}\n"
            result_text += f"Experience: {', '.join(experience) if experience else 'None found'}\n"

            st.markdown(create_download_link(result_text), unsafe_allow_html=True)
