# 📄 AI Resume Screening & Candidate Ranking System  

This application automates resume screening using **AI-driven text analysis**. It leverages **TF-IDF vectorization** and **cosine similarity** to compare candidate resumes against a job description and rank them based on relevance.  

## 🚀 Features  

✅ **Upload Multiple Resumes** (PDF format)  
✅ **Extract Text** from resumes using `PyPDF2`  
✅ **Compare Resumes** with job descriptions using **TF-IDF**  
✅ **Rank Candidates** based on cosine similarity score  
✅ **Interactive UI** built with **Streamlit**  

## 🛠️ Tech Stack  

- **Python**  
- **Streamlit** (for UI)  
- **PyPDF2** (for PDF parsing)  
- **Scikit-learn** (for text processing & similarity scoring)  
- **Pandas** (for result display)  

## 📥 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/resume-screening-ai.git
   cd resume-screening-ai
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

## 📌 Usage  

1. Enter the **job description** in the text area.  
2. Upload multiple **PDF resumes**.  
3. The system will extract text, compare resumes, and **rank candidates**.  
4. View ranked results based on **cosine similarity scores**.  
 

## 🤝 Contributing  

Feel free to **fork** this repo, submit **issues**, or create **pull requests** for improvements!  

