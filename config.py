import os

# Paths
PRELOADED_RESOURCES = {
    "Finance CSV (local_data_sources/finance.csv)": "local_data_sources/finance.csv",
    "Finance PDF (local_data_sources/pfi-briefings.pdf)": "local_data_sources/pfi-briefings.pdf",
}
LLAMA_MODEL_PATH = "./llm_model/gemma-3-4b-it-UD-Q4_K_XL.gguf"
PERSIST_DIR = "chroma_db"

# LLM Params
LLM_PARAMS = dict(
    temperature=0.1,
    max_tokens=256,
    n_gpu_layers=int(os.environ.get("N_GPU_LAYERS", "0")),
    n_batch=256,
    n_ctx=8192,
    verbose=False,
)

# Prompt
PROMPT_TEMPLATE = """
You are a helpful financial assistant.
Use ONLY the information in the CONTEXT below to answer the QUESTION.
If the answer isn't in the context, reply exactly: "I dont know based on the provided document."
Cite each chunk you use, e.g. [Chunk 2].

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

# Example questions
EXAMPLE_QUESTIONS = [
    "What are the main concepts of finance?",
    "What is the time value of money?",
    "How do financial markets function?",
    "What are financial instruments?",
    "What is the relationship between risk and return?"
]

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
