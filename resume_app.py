import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 


def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

#calculate cosine similarity

    job_description_vector = vectors[0]
    resume_vectors =  vectors[1:]     #slicing operation -> 1 onwards
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities 

# close to 1 -> accurately matching
# close to 0 -> not matching

#streamlit_App
st.title("AI Resume Screening & Candidate Ranking System")

# job description input
st.header("Job Description")
job_description = st.text_area("Enter the Job Description")


#file uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type = ["pdf"], accept_multiple_files = True)
if uploaded_files and job_description:
    st.header("Ranking Resumes")
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    scores = rank_resumes(job_description, resumes)

    result = pd.DataFrame({"Resume" : [file.name for file in uploaded_files], "Score": scores})
    results = result.sort_values(by = "Score", ascending = False)
    st.write(results)