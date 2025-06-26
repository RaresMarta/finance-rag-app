# 📊 Finance Knowledge RAG Demo

A Retrieval-Augmented Generation (RAG) app using LangChain and LLama.cpp to answer finance-related questions from uploaded documents (PDF, CSV, DOCX, TXT).

## 🔧 Features
- Upload or select preloaded files
- Chunk, embed, and persist data with ChromaDB
- Ask natural-language questions
- Use MMR or Similarity-based retrieval
- LLama.cpp local inference

## 📁 File Structure
| File / Folder         | Description                               |
|----------------------|-------------------------------------------|
| `main.py`            | Full app logic                            |
| `config.py`          | Config: model path, chunk size, prompt    |
| `requirements.txt`   | Python dependencies                       |
| `Dockerfile`         | Build instructions for Docker             |
| `local_data_sources/`| Example documents (e.g., finance.csv)     |
| `.dockerignore`      | Files ignored when building Docker image  |
| `.gitignore`         | Files ignored by Git                      |

## 🚀 Getting Started

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## 🐳 Docker

Build the image:
```bash
docker build -t finance-rag-app .
```

Run the app:
```bash
docker run -p 8501:8501 finance-rag-app
```

> **Note:**  
> This setup does **not** persist embeddings, models, or uploaded data between runs.  
> Each time you start the container, embeddings will be recomputed as needed.

