import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

st.set_page_config(page_title="Equity Research Analyst", page_icon="📊", layout="wide")

st.title("🧠 Equity Research Analyst")
st.markdown("Analyze financial or news articles using your local AI model (no API key needed).")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Data Input")

urls = st.sidebar.text_area(
    label="Enter article URLs (one per line):",
    placeholder="https://www.reuters.com/markets/example-article\nhttps://finance.yahoo.com/news/example",
    height=150,
)

process_btn = st.sidebar.button("🚀 Process Articles")

# -------------------------
# Define constants
# -------------------------
DB_FAISS_PATH = "faiss_index"

# -------------------------
# Step 1: Load Data
# -------------------------
if process_btn:
    if not urls.strip():
        st.warning("⚠️ Please provide at least one valid URL.")
    else:
        with st.spinner("Loading and processing URLs..."):
            loader = WebBaseLoader(urls)
            data = loader.load()
            
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(data)

            # Create embeddings and FAISS index
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(DB_FAISS_PATH)

            st.success("✅ FAISS index created and saved!")

# -------------------------
# Step 2: Ask Questions
# -------------------------
st.markdown("---")
st.header("💬 Ask Questions About the Data")

question = st.text_input(
    "Enter your question about the content:",
    placeholder="e.g., What are the main factors affecting Tesla’s earnings?"
)
ask_btn = st.button("🔍 Analyze")

if ask_btn:
    if not os.path.exists(DB_FAISS_PATH):
        st.error("⚠️ Please process some articles first.")
    else:
        with st.spinner("Generating response..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

            # Use local Llama model via Ollama
            llm = Ollama(model="llama3", temperature=0.7)

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(),
                return_source_documents=True,
            )

            result = qa.invoke({"query": question})

            st.subheader("📊 Answer")
            st.write(result.get("result", "No answer found."))

            with st.expander("🔗 Source Documents"):
                for doc in result.get("source_documents", []):
                    st.markdown(f"- {doc.metadata.get('source', 'Unknown source')}")

st.markdown("---")
st.caption("Developed by Ishaan Farooq – Powered by LangChain, FAISS, and Llama")
