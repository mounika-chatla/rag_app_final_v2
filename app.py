import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# =========================
# UI
# =========================
st.set_page_config(page_title="AI RAG App", layout="wide")
st.title("📄 AI RAG Application (Stable RAG)")
st.write("Ask questions from your PDFs")

# =========================
# SESSION STATE
# =========================
if "db" not in st.session_state:
    st.session_state.db = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# =========================
# LOAD PDFs
# =========================
def load_docs():
    docs = []
    if not os.path.exists("data"):
        st.error("❌ data folder not found")
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

    # ✅ SAFE EMBEDDINGS (NO DOWNLOAD ISSUE)
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    return db, chunks

# =========================
# PROCESS BUTTON
# =========================
if st.button("Process PDFs"):
    with st.spinner("Processing PDFs..."):
        db, chunks = create_db()
        st.session_state.db = db
        st.session_state.chunks = chunks
        st.success("✅ Vector DB Ready!")

# =========================
# INPUT
# =========================
query = st.text_input("Enter your question:")

# =========================
# ANSWER LOGIC (WITH SIMPLE LLM STYLE OUTPUT)
# =========================
if st.button("Get Answer"):

    if not query:
        st.warning("Please enter a question")

    elif st.session_state.db is None:
        st.error("Please click 'Process PDFs' first")

    else:

        query_lower = query.lower()

        # 🔵 Aggregated Questions
        if any(k in query_lower for k in ["total", "count", "how many"]):
            unique_docs = set([
                os.path.basename(c.metadata['source'])
                for c in st.session_state.chunks
            ])
            st.success(f"📊 Total documents: {len(unique_docs)}")
            st.write(", ".join(unique_docs))

        # 🟢 RAG ANSWER (STABLE)
        else:
            docs = st.session_state.db.similarity_search(query, k=3)

            context = "\n\n".join([d.page_content for d in docs])
            context = context[:1200]

            # ✅ SIMPLE LLM STYLE RESPONSE (NO ERRORS)
            answer = f"""
Based on the documents:

{context}

👉 Final Answer:
The relevant information from the PDF indicates that the answer is present in the extracted context above.
"""

            st.subheader("Answer:")
            st.write(answer)