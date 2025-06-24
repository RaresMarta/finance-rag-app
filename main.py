import os
import tempfile
import hashlib
from pathlib import Path
import streamlit as st
import requests
import certifi

from config import (
    PRELOADED_RESOURCES, LLAMA_MODEL_PATH, PERSIST_DIR,
    LLM_PARAMS, PROMPT_TEMPLATE, EXAMPLE_QUESTIONS,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# LangChain / community imports
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Streamlit setup
st.set_page_config(page_title="Finance RAG Demo", layout="wide")
st.title("Finance Knowledge RAG Demo")

# Sidebar: select built-in resource
selected_source = st.sidebar.selectbox(
    "Select Resource:", list(PRELOADED_RESOURCES.keys())
)

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Or upload a document (PDF, CSV, TXT, DOCX):",
    type=["pdf", "csv", "txt", "docx"],
)

if uploaded_file:
    st.sidebar.info(f"Using uploaded file: {uploaded_file.name}")
else:
    st.sidebar.info(f"Using preloaded: {selected_source}")

# Compute hash of uploaded file (if any)
uploaded_hash = None
if uploaded_file:
    data_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    uploaded_hash = hashlib.md5(data_bytes).hexdigest()

# Determine resource ID & path
if uploaded_file:
    resource_id = uploaded_hash
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data_bytes)
    tmp_path = tmp.name
    tmp.close()
    resource_path = tmp_path
else:
    src = PRELOADED_RESOURCES[selected_source]
    resource_id = hashlib.md5(str(src).encode()).hexdigest()
    resource_path = src

# Persist directory
Path(PERSIST_DIR).mkdir(exist_ok=True)
resource_persist_dir = Path(PERSIST_DIR) / resource_id

# Check rebuild: new upload or empty dir or forced
prev_hash = st.session_state.get("last_upload_hash")
need_hash_rebuild = uploaded_file and (uploaded_hash != prev_hash)
force_rebuild = st.sidebar.button("Force Rebuild Embeddings")
rebuild_needed = (
    need_hash_rebuild
    or force_rebuild
    or not resource_persist_dir.exists()
    or not any(resource_persist_dir.iterdir())
)
if uploaded_file:
    st.session_state["last_upload_hash"] = uploaded_hash

# Embeddings & LLM factory
embeddings = GPT4AllEmbeddings()

@st.cache_resource
def get_llm():
    return LlamaCpp(
        model_path=LLAMA_MODEL_PATH,
        **LLM_PARAMS
    )

# Prompt template for direct answers
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)

# Build or load embeddings
if rebuild_needed:
    # 1) Load data
    with st.spinner("Loading document/dataset…"):
        suffix = Path(resource_path).suffix.lower()
        if suffix == ".pdf" and str(resource_path).startswith("http"):
            resp = requests.get(resource_path, verify=certifi.where())
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(resp.content)
            path = tmp.name; tmp.close()
            data = PyPDFLoader(path).load()
            os.unlink(path)
        elif suffix == ".pdf":
            data = PyPDFLoader(resource_path).load()
        elif suffix == ".csv":
            data = CSVLoader(resource_path).load()
        elif suffix == ".txt":
            data = TextLoader(resource_path).load()
        elif suffix == ".docx":
            data = Docx2txtLoader(resource_path).load()
        else:
            st.error(f"Unsupported file type: {suffix}")
            st.stop()

    # 2) Chunk
    with st.spinner("Splitting into chunks…"):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
        )
        chunks = splitter.split_documents(data)
        st.success(f"Split into {len(chunks)} chunks")

    # 3) Embed & persist
    with st.spinner("Creating vectorstore…"):
        if resource_persist_dir.exists():
            for f in resource_persist_dir.iterdir():
                f.unlink()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(resource_persist_dir),
        )
        vectorstore.persist()
        st.success("Embeddings saved!")
else:
    with st.spinner("Loading existing embeddings…"):
        vectorstore = Chroma(
            persist_directory=str(resource_persist_dir),
            embedding_function=embeddings,
        )
        st.success("Loaded embeddings.")

# ——— QA Interface ———
if "example_idx" not in st.session_state:
    st.session_state.example_idx = 0

st.header("Ask a Finance Question")
question = st.text_input("Enter your question:", st.session_state.get("question", ""))

if st.button("Use Example Question"):
    idx = st.session_state.example_idx
    question = EXAMPLE_QUESTIONS[idx]
    st.session_state.question = question
    st.session_state.example_idx = (idx + 1) % len(EXAMPLE_QUESTIONS)
    st.rerun()

retrieval_method = st.radio("Retrieval Method:", ["Similarity", "MMR (Diverse)"])
num_chunks = st.slider("Number of chunks to retrieve:", 1, 5, 3)

if st.button("Get Answer"):
    with st.spinner("Retrieving…"):
        if retrieval_method == "MMR (Diverse)":
            docs = vectorstore.max_marginal_relevance_search(
                query=question, k=num_chunks, fetch_k=num_chunks*3, lambda_mult=0.7
            )
        else:
            docs = vectorstore.similarity_search(question, k=num_chunks)

    unique, seen = [], set()
    for i, d in enumerate(docs):
        if d.page_content not in seen:
            unique.append((i, d.page_content))
            seen.add(d.page_content)

    context = "\n\n".join(f"[Chunk {i+1}]\n{text}" for i, text in unique)

    llm = get_llm()
    qa = LLMChain(llm=llm, prompt=PROMPT)
    answer = qa.run(context=context, question=question)

    if "dont know" in answer:
        st.info("The model could not find an answer in the provided document.")

    with st.expander("Context Chunks", expanded=True):
        for i, text in unique:
            st.markdown(f"**Chunk {i+1}**")
            st.write(text)
            st.markdown("---")

    st.markdown("### Answer")
    st.write(answer)
