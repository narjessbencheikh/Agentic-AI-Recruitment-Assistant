Voilà le README complet et pro :
markdown# 🤖 AI Recruitment Assistant

An intelligent recruitment assistant powered by **Generative AI** and **Agentic AI** that analyzes job descriptions and generates optimized CVs based on required skills.

---

## 🎯 Overview

This system combines **Multi-Agent AI** and **RAG (Retrieval-Augmented Generation)** to:
1. Automatically analyze a job description
2. Extract key skills and selection criteria
3. Search and benchmark similar successful profiles
4. Generate an optimized, ATS-friendly CV

---

## 🏗️ Architecture

### Part 1 — Multi-Agent System (LangGraph)
Job Description (input)

↓

Orchestrator (LangGraph)

↙            ↘

Job Analyzer   Skill Extractor

↓

Profile Searcher

↓

Structured Analysis (output)

**Agents:**
- **JobAnalyzerAgent** → extracts job title, sector, experience level and responsibilities
- **SkillExtractorAgent** → categorizes must-have, nice-to-have and soft skills
- **ProfileSearcherAgent** → builds an ideal benchmark profile for the position
- **OrchestratorAgent** → coordinates all agents via LangGraph StateGraph with error handling

### Part 2 — RAG System
data/cvs/ (PDFs)

↓

CVIndexer → ChromaDB (vector store)

↓

CVRetriever → Top-K similar profiles

↓

CVGenerator → Optimized CV (Mistral 7B)

**Pipeline:**
- **Indexing** → PDFs extracted, chunked (500 words, 50 overlap), embedded with `all-MiniLM-L6-v2`
- **Retrieval** → semantic similarity search with cosine distance
- **Generation** → contextualized prompt + Mistral 7B via Ollama

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Mistral 7B via Ollama |
| Agents | LangGraph + LangChain |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF Parsing | pypdf |
| UI | Streamlit |
| Containerization | Docker |

---

## 📁 Project Structure
ai-recruitment-assistant/

├── agents/

│   ├── init.py

│   ├── orchestrator.py      # LangGraph orchestration

│   ├── job_analyzer.py      # Job description analysis

│   ├── skill_extractor.py   # Skills extraction

│   └── profile_searcher.py  # Benchmark profile generation

├── rag/

│   ├── init.py

│   ├── indexer.py           # PDF indexing into ChromaDB

│   ├── retriever.py         # Semantic search

│   ├── generator.py         # CV generation

│   └── prompt_templates.py  # Centralized prompts

├── evaluation/

│   ├── init.py

│   └── metrics.py           # RAG evaluation metrics

├── data/

│   ├── cvs/                 # CV database (PDFs)

│   └── job_descriptions/    # Job descriptions (PDFs)

├── app.py                   # Streamlit interface

├── .env                     # Environment variables

├── requirements.txt

├── Dockerfile

└── docker-compose.yml

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Ollama installed with Mistral 7B

### 1. Install Mistral via Ollama
```bash
ollama pull mistral
```

### 2. Clone the repository
```bash
git clone https://github.com/narjessbencheikh/AI-Recruitment-Assistant.git
cd AI-Recruitment-Assistant
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
Create a `.env` file :
```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=mistral
```

### 5. Add your CVs
Put your PDF CVs in `data/cvs/`

### 6. Run the app
```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

```bash
docker-compose up
```

Access the app at `http://localhost:8501`

---

## 💻 Usage

1. **Index CVs** → click "Index CVs" in the sidebar (do this once)
2. **Tab 1 - Job Analysis** → paste a job description and click Analyze
3. **Tab 2 - CV Generator** → enter candidate profile and generate optimized CV
4. **Tab 3 - Evaluation** → run metrics to evaluate the system quality

---

## 📊 Evaluation Metrics

### Retrieval Metrics

| Metric | Description | Formula |
|---|---|---|
| Precision@K | % of retrieved CVs that are relevant | relevant retrieved / K |
| Recall@K | % of relevant CVs that were retrieved | relevant retrieved / total relevant |
| MRR | Is the best result ranked first? | 1 / position of first relevant result |

### Generation Metrics

| Metric | Description | Formula |
|---|---|---|
| Relevance Score | Semantic similarity between CV and job | Cosine similarity (embeddings) |
| Skill Match Score | % of required skills present in generated CV | matched skills / total required |

### Run evaluation
```bash
python -c "from evaluation.metrics import RAGEvaluator; print(RAGEvaluator().evaluate(query='Python machine learning', relevant_sources=['cv1.pdf'], generated_cv='...', job_description='...', required_skills=['Python']))"
```

---

## 🔧 Optimization & Improvements

### Current Limitations
1. **Low retrieval scores with small CV database** → improve by adding more CVs
2. **Mistral sometimes returns malformed JSON** → handled with regex cleaning
3. **No persistent session state** → CVs must be re-indexed on restart

### Improvement Strategies
1. Add **hybrid search** (semantic + keyword) for better retrieval
2. Fine-tune prompts with **few-shot examples** for more consistent JSON output
3. Add **persistent vector store** with automatic re-indexing

---

## 👤 Author

**Narjess Bencheikh**  
