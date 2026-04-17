import streamlit as st
import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# =========================
# GROQ LLM (FIXED)
# =========================
client = Groq(api_key=st.secrets["mounika123"])

def ask_llm(context, question):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the given context."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}

Answer clearly and short:
"""
            }
        ]
    )
    return response.choices[0].message.content

# =========================
# UI
# =========================
st.set_page_config(page_title="RAG + LLM App", layout="wide")
st.title("📄 AI RAG Application (FAISS + GROQ LLM)")
st.write("Ask questions from your PDFs")

# =========================
# STATE
# =========================
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LOAD PDFS
# =========================
def load_docs():
    docs = []
    if not os.path.exists("data"):
        return docs

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            docs.extend(loader.load())

    return docs

# =========================
# CREATE VECTOR DB
# =========================
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

# =========================
# PROCESS PDFs
# =========================
if st.button("Process PDFs"):
    with st.spinner("Creating Vector DB..."):
        index, chunks = create_db()
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success("✅ Vector DB Ready")

# =========================
# QUERY
# =========================
query = st.text_input("Enter your question:")

if st.button("Get Answer"):

    if not query:
        st.warning("Enter question")

    elif st.session_state.index is None:
        st.error("Click Process PDFs first")

    else:

        q_emb = model.encode([query])

        D, I = st.session_state.index.search(
            np.array(q_emb),
            k=3
        )

        docs = [st.session_state.chunks[i] for i in I[0]]
        context = "\n\n".join([d.page_content for d in docs])

        if any(x in query.lower() for x in ["count", "how many", "total"]):
            answer = f"Total documents used: {len(set([d.metadata['source'] for d in docs]))}"
        else:
            answer = ask_llm(context[:2000], query)

        st.subheader("Answer:")
        st.success(answer)