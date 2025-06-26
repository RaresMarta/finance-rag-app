# 📊 Finance Knowledge RAG Demo

A Retrieval-Augmented Generation (RAG) app using LangChain and LLama.cpp to answer finance-related questions from uploaded documents (PDF, CSV, DOCX, TXT).

## 🔧 Features
- Upload or select preloaded files
- Chunk, embed, and persist data with ChromaDB
- Ask natural-language questions
- Use MMR or Similarity-based retrieval
- LLama.cpp local inference

## 📁 File Structure
- `main.py`: Streamlit front-end
- `config.py`: Configuration constants
- `local_data_sources/`: Example documents
- `llm_model/`: Path for LLaMA model (excluded from repo)
- `chroma_db/`: Chroma DB (excluded from repo)

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
> Each time you start the container, embeddings will be recomputed as

