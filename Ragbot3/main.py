# streamlit run main.py
import streamlit as st
import os
from datetime import datetime
import time
import hashlib
from pathlib import Path
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# Streamlit Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="centered")

st.title("📚 Document Q&A System")
st.caption("Ask questions from your uploaded documents using local AI models (LangChain + ChromaDB)")

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
session_defaults = {
    "messages": [],
    "vectorstore": None,
    "documents": [],
    "embeddings": None,
    "rag_chain": None,
    "retriever": None,
    "response_times": [],
    "cache": {},
    "total_chunks": 0,
    "llm_loaded": False,
    "llm": None
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
LLM_MODEL = "google/flan-t5-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
K_DOCUMENTS = 3
VECTOR_DIR = "./chroma_db"

PROMPT_TEMPLATE = """Answer the question based on the following context. 
If you cannot answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

def extract_answer(raw):
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            return first.get("generated_text") or first.get("text") or str(first)
        return str(first)
    if isinstance(raw, dict):
        return raw.get("generated_text") or raw.get("text") or str(raw)
    return str(raw)

@st.cache_resource
def load_local_llm(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", dtype=torch.float32)
        except Exception:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to("cpu")

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, do_sample=False)
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def initialize_vectorstore():
    if st.session_state.embeddings is None:
        return None
    try:
        return Chroma.from_documents([], st.session_state.embeddings, persist_directory=VECTOR_DIR)
    except Exception:
        return None

def create_rag_chain(vectorstore):
    if not vectorstore or st.session_state.llm is None:
        return None
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_DOCUMENTS})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    st.session_state.retriever = retriever
    st.session_state.rag_chain = rag_chain
    return rag_chain

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
if not st.session_state.llm_loaded:
    with st.spinner("Loading local model..."):
        llm = load_local_llm(LLM_MODEL)
        if llm:
            st.session_state.llm = llm
            st.session_state.llm_loaded = True
            st.success("Model loaded successfully!")
            time.sleep(0.5)
            st.rerun()

# ─────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────
st.subheader("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, TXT, or DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)
if st.button("Process Documents", use_container_width=True) and uploaded_files:
    with st.spinner("Processing documents..."):
        try:
            if st.session_state.embeddings is None:
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = (
                    PyPDFLoader(tmp_path) if uploaded_file.name.endswith(".pdf")
                    else TextLoader(tmp_path) if uploaded_file.name.endswith(".txt")
                    else Docx2txtLoader(tmp_path)
                )
                docs = loader.load()
                all_docs.extend(docs)
                os.unlink(tmp_path)

            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents(all_docs)
            st.session_state.total_chunks = len(splits)

            vectorstore = Chroma.from_documents(splits, st.session_state.embeddings, persist_directory=VECTOR_DIR)
            st.session_state.vectorstore = vectorstore
            create_rag_chain(vectorstore)

            st.success(f"Processed {len(uploaded_files)} files ({len(splits)} chunks)")
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# Chat Section
# ─────────────────────────────────────────────
st.subheader("💬 Ask Your Question")
if st.session_state.rag_chain:
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        user_query = st.text_input("Type your question:", placeholder="Ask about your documents...")
    with col_btn:
        send = st.button("Send")

    use_cache = st.checkbox("Use cached responses", value=True)

    if send and user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        cache_key = hashlib.md5(user_query.lower().encode()).hexdigest()

        if use_cache and cache_key in st.session_state.cache:
            answer = f"(cached) {st.session_state.cache[cache_key]}"
        else:
            with st.spinner("Generating answer..."):
                start = time.time()
                retriever = st.session_state.retriever
                retrieved = retriever.invoke(user_query)
                context_text = format_docs(retrieved)
                prompt_text = PROMPT_TEMPLATE.format(context=context_text, question=user_query)
                raw = st.session_state.llm.invoke(prompt_text)
                answer = extract_answer(raw)
                st.session_state.cache[cache_key] = answer
                st.session_state.response_times.append(time.time() - start)

        st.markdown("### 🤖 Answer")
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload and process your documents first.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("RAG Q&A System • Built with LangChain + ChromaDB • Local Inference")
