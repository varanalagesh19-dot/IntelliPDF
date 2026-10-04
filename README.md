---
title: IntelliPDF
emoji: 📘
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI-powered exam prep from any PDF
---

# IntelliPDF

Upload any subject PDF → get mark-wise answers (1M–16M), auto quizzes, study plans, weak-topic detection, and mock exams — all grounded in your own notes with page references.

## Features

- Upload any PDF — ML, NLP, DBMS, Networks, or any subject
- Mark-wise answers — 1M, 2M, 5M, 10M, 15M, 16M
- RAG-based Q&A grounded in your PDF with page references
- Auto-generated quizzes with weak-topic detection
- AI study plan based on exam date and hours
- Answer evaluation with missing points
- Mock exam mode with timer
- Multilingual — English, Tamil, Tanglish
- Source verification with page citations

## Tech Stack

FastAPI-style services · Streamlit UI · PyMuPDF · LangChain chunking · sentence-transformers · FAISS · Groq / Gemini / Ollama fallback · SQLite

## Getting Started (local)

    git clone https://huggingface.co/spaces/Atlascore77/IntelliPDF
    cd IntelliPDF
    python -m venv venv
    venv\Scripts\activate     # Windows
    pip install -r requirements.txt
    # Add GROQ_API_KEY to .env
    streamlit run app.py

## License

MIT