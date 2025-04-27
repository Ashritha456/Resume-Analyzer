# 🧠 AI Resume Analyzer + Job Recommender

A **smart AI Resume Analyzer + Job Recommender** that extracts key information from resumes (skills, education, experience), predicts the candidate's domain using keyword clustering, detects missing resume sections, scores the resume out of 100, recommends courses, and optionally uses OpenAI to suggest resume improvements.

---

## 📜 Features

| Feature | Description |
|:--------|:------------|
| 📄 Resume Upload | Upload resumes in PDF or TXT format |
| 🔍 Resume Parsing | Extract Skills, Education, and Experience |
| 🧠 Domain Prediction | Predict candidate’s domain (e.g., Cloud/DevOps, Data Science) |
| 🎯 Personalized Feedback | Identify skill gaps, missing sections, and suggest courses |
| 📈 Resume Scoring | Score resume quality out of 100 |
| 📑 Missing Sections Detector | Warn if "Projects", "Certifications", etc. are missing |
| 🤖 AI-Powered Suggestions | (Optional) Use OpenAI GPT-4 to suggest resume improvements |
| 📥 Download Report | Download a full analysis report as a `.txt` file |

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) — Interactive Web UI
- [Huggingface Transformers](https://huggingface.co/transformers/) — BERT Resume Classifier
- [PyTorch](https://pytorch.org/) — Model Inference
- [PyPDF2](https://pypi.org/project/PyPDF2/) — Resume PDF parsing
- [Altair](https://altair-viz.github.io/) — Data visualization
- [OpenAI GPT-4 API](https://platform.openai.com/) — (Optional) Resume enhancement suggestions

---

## 📂 Project Structure

```
  ├── app.py                     # Main Streamlit app
                
 ├── config.json
 ├── pytorch_model.bin
 ├── tokenizer_config.json
 ├── special_tokens_map.json
 └── vocab.txt
 ├── README.md                   # This file
 └── requirements.txt            # Python dependencies
```

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/smart-resume-analyzer.git
cd smart-resume-analyzer
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
torch
transformers
PyPDF2
pandas
altair
openai
```

3. **Run the Streamlit App:**

```bash
streamlit run app.py
```

---

## 🔑 OpenAI API Key (Optional)

- To enable **AI Suggestions**, create a free [OpenAI account](https://platform.openai.com/signup).
- Generate an API key and paste it into the input field when prompted inside the app.

*(No key needed for basic resume parsing, prediction, scoring, etc.)*

---

## 📥 Downloadable Analysis

After analysis, you can **download a `.txt` report** containing:
- Predicted job role
- Domain detected
- Skills and missing skills
- Resume Score
- Resume Tips
- Course recommendations
- (Optional) GPT-4 AI feedback

---

## 🎯 Future Improvements

- Bulk resume upload (analyze 100 resumes at once)
- Advanced keyword extraction using Named Entity Recognition (NER)
- Generate resume improvement recommendations visually
- Fine-tuned BERT with larger dataset
- PDF Report generation

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

MIT License © [Your Name]

---

## ✨ Special Thanks

- [Streamlit Team](https://streamlit.io/)
- [Huggingface](https://huggingface.co/)
- [OpenAI](https://openai.com/)

---

## ✅ Quick Commands Summary

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📸 Example App Flow

> Upload resume → View parsed data → Predict role/domain → Get skill gaps & resume tips → (Optional) AI improvement suggestions → Download report

---

