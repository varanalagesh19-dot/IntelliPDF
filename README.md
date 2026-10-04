# IntelliPDF

> AI-powered personalized exam learning assistant that converts study materials into mark-wise preparation, practice, evaluation, and revision content.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- Upload any PDF — ML, NLP, DBMS, Networks, or any subject
- Mark-wise answers — 1M, 2M, 5M, 10M, 15M, 16M
- RAG-based Q&A grounded in your PDF with page references
- Auto-generated quizzes with scoring and weak-topic detection
- AI study plan based on exam date and daily hours
- Answer evaluation — find missing points vs source material
- Mock exam mode with timed tests and mixed marks
- Multilingual — English, Tamil, Tanglish
- Last-minute revision — flashcards and key concepts
- Source verification — clickable page references

## Architecture

    ┌─────────────────────────────────────────────────────────┐
    │              Streamlit Frontend (Port 8501)             │
    │  Upload │ Ask │ Quiz │ Exam │ Study Plan │ Revision     │
    └────────────────────────┬────────────────────────────────┘
                             │ HTTP
    ┌────────────────────────▼────────────────────────────────┐
    │              FastAPI Backend (Port 8000)                │
    │  /upload │ /ask │ /quiz │ /mock-exam │ /study-plan      │
    └────────────────────────┬────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
       ┌─────────┐      ┌─────────┐     ┌──────────┐
       │ PyMuPDF │      │ Groq /  │     │  FAISS   │
       │  (PDF)  │      │ Gemini  │     │ (Vectors)│
       └─────────┘      └─────────┘     └──────────┘
                             │
                    ┌────────▼────────┐
                    │ SQLite (History)│
                    └─────────────────┘

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| LLM | Groq -> Gemini -> Ollama (fallback chain) |
| Storage | SQLite |
| Evaluation | SQuAD 2.0 |

## Quick Start

### 1. Clone the repository
    git clone https://github.com/YOUR_USERNAME/IntelliPDF.git
    cd IntelliPDF

### 2. Create virtual environment
    python -m venv venv
    venv\Scripts\activate          # Windows
    # source venv/bin/activate     # Mac/Linux

### 3. Install dependencies
    pip install -r requirements.txt

### 4. Get free API keys
- Groq (recommended, fast): https://console.groq.com/keys
- Gemini (fallback): https://aistudio.google.com/apikey
- Ollama (offline): https://ollama.com

### 5. Configure environment
    cp .env.example .env
    # Edit .env and add your GROQ_API_KEY (and GEMINI_API_KEY if using)

### 6. Run the app
    # Terminal 1 - Backend
    uvicorn backend.main:app --reload --port 8000

    # Terminal 2 - Frontend
    streamlit run frontend/app.py

Open http://localhost:8501

## Project Structure

    intellipdf/
    ├── backend/
    │   ├── main.py
    │   ├── config.py
    │   ├── routers/          # API endpoints
    │   ├── services/         # Core logic (PDF, RAG, LLM, etc.)
    │   ├── models/           # Pydantic schemas
    │   ├── utils/            # DB, language, metadata helpers
    │   └── data/             # Uploads + vector DB (gitignored)
    ├── frontend/
    │   └── app.py            # Streamlit UI
    ├── evaluation/
    │   └── squad_eval.py     # SQuAD 2.0 evaluation
    ├── tests/
    │   └── test_pipeline.py
    ├── requirements.txt
    ├── .env.example
    ├── run.sh
    ├── run.bat
    └── README.md

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/upload | Upload and process PDF |
| POST | /api/ask | Ask a question (mark-wise) |
| POST | /api/quiz | Generate quiz |
| POST | /api/evaluate | Evaluate student answer |
| POST | /api/study-plan | Generate study plan |
| POST | /api/mock-exam | Start timed mock exam |
| GET | /api/weak-topics/{doc_id} | Get weak topics |
| GET | /health | Health check |

## References

- Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text," EMNLP 2016
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020
- Reimers and Gurevych, "Sentence-BERT," EMNLP-IJCNLP 2019
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- BE CSE / AIML Project
- Built with FastAPI, Streamlit, LangChain, and free-tier LLMs