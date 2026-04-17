import streamlit as st
import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# -------------------------
# SAFE SECRET CHECK
# -------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# -------------------------
# LLM FUNCTION
# -------------------------
def ask_llm(context, question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer only from context."},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )
    return response.choices[0].message.content

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 AI RAG Application")

# -------------------------
# STATE
# -------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# LOAD DOCS
# -------------------------
def load_docs():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            docs.extend(loader.load())
    return docs

# -------------------------
# VECTOR DB
# -------------------------
def create_db():
    docs = load_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return index, chunks

# -------------------------
# PROCESS
# -------------------------
if st.button("Process PDFs"):
    index, chunks = create_db()
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.success("✅ Ready")

# -------------------------
# QUERY
# -------------------------
query = st.text_input("Ask question")

if st.button("Get Answer"):

    q_emb = model.encode([query])

    D, I = st.session_state.index.search(np.array(q_emb), k=3)

    docs = [st.session_state.chunks[i] for i in I[0]]
    context = "\n\n".join([d.page_content for d in docs])

    answer = ask_llm(context[:2000], query)

    st.success(answer)