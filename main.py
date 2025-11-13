

import os
import re
import uuid
import subprocess
import glob
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

import whisper

# -----------------------
# Environment + Page Setup
# -----------------------
load_dotenv()
st.set_page_config(page_title="Equity Research Analyst", page_icon="📊", layout="wide")

# -----------------------
# Glassmorphism CSS (paste right after page config)
# -----------------------
glass_css = """
<style>

body {
    background: linear-gradient(135deg, #071126 0%, #0f1536 100%);
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    color: #e6eef8;
}

/* Layout padding */
.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Gradient title */
h1 {
    text-align: center;
    font-size: 2.6rem !important;
    background: linear-gradient(90deg, #7af7d3, #8b74ff 60%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 0.1rem;
}

/* Subtitle */
.app-sub {
    text-align:center;
    color: #b9c6df;
    margin-bottom: 1.4rem;
    font-size: 1.02rem;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    padding: 22px 26px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(12px) saturate(120%);
    box-shadow: 0 12px 30px rgba(2,6,23,0.6);
    margin-bottom: 18px;
}

/* Sidebar glass */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
    border-right: 1px solid rgba(255,255,255,0.04);
    backdrop-filter: blur(18px);
}

/* Inputs */
textarea, .stTextInput > div > div > input, .stSelectbox > div > div {
    background: rgba(255,255,255,0.035) !important;
    color: #e6eef8 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    padding: 10px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg,#7a66ff,#6ef0d4);
    color: #031028 !important;
    border-radius: 12px;
    font-weight: 700;
    padding: 8px 14px;
    border: none;
    box-shadow: 0 8px 20px rgba(106,90,255,0.18);
    transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button:hover { transform: translateY(-3px) scale(1.02); box-shadow: 0 14px 30px rgba(106,90,255,0.22); }

/* Answer box */
.answer-box {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.05);
    backdrop-filter: blur(8px);
    color: #dbeafe;
}

/* Source list */
.source-list { color: #bcd2ff; }

/* Expander style */
.streamlit-expanderHeader {
    color: #7af7d3;
    font-weight: 700;
}

/* Small muted text */
.small-muted { color: #9fb0d6; font-size: 0.92rem; }

</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)

# -----------------------
# App title + subtitle (in glass container)
# -----------------------
st.title("🧠 Equity Research Analyst")
st.markdown("<div class='app-sub'>Analyze articles, financial news & YouTube videos locally using Whisper + Llama.</div>", unsafe_allow_html=True)

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("Data Input")

urls = st.sidebar.text_area(
    label="Enter article or YouTube URLs (one per line):",
    placeholder="https://www.reuters.com/markets/example-article\nhttps://www.youtube.com/watch?v=IlbyURD_e7c",
    height=160,
)

process_btn = st.sidebar.button("🚀 Process Sources")
DB_FAISS_PATH = "faiss_index"

# -----------------------
# Helper: Playwright article loader
# -----------------------
def load_url_with_playwright(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            selectors = [
                "article",
                ".entry-content",
                ".post-content",
                ".td-post-content",
                ".content",
                "#content",
            ]

            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=7000)
                    break
                except:
                    continue

            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "html.parser")
        article = None
        for sel in selectors:
            block = soup.select_one(sel)
            if block:
                article = block
                break

        target = article if article else soup
        for tag in target(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        text = target.get_text(separator="\n", strip=True)
        return text

    except Exception as e:
        return f"ERROR loading article: {str(e)}"

# -----------------------
# Helper: Whisper + yt-dlp transcription (uses unique temp files)
# -----------------------
def whisper_transcribe_youtube(url):
    """
    Downloads audio via yt-dlp into a unique temp WAV file and transcribes with Whisper.
    Returns text or an ERROR string starting with 'ERROR'.
    """
    unique_id = uuid.uuid4().hex
    tmp_wav = f"tmp_{unique_id}.wav"

    # Build yt-dlp command that produces WAV output directly
    # Using -x and --audio-format wav and output as tmp_wav
    try:
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "-x",
            "--audio-format", "wav",
            "-o", tmp_wav,
            url
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        # Try without suppressing output to give more debug info
        try:
            subprocess.run(cmd, check=True)
        except Exception as e2:
            return f"ERROR downloading YouTube audio: {str(e2)}"

    # After download, yt-dlp should have created a file like tmp_<id>.wav
    matched = glob.glob(f"tmp_{unique_id}.*")
    if not matched:
        # fallback: try common alternative pattern
        matched = glob.glob(f"tmp_{unique_id}*")
    if not matched:
        return "ERROR: audio file not found after yt-dlp."

    audio_file = matched[0]

    # Ensure extension is .wav; if not, convert using ffmpeg via yt-dlp already asked for wav,
    # but handle safe fallback: convert to wav using ffmpeg if needed
    if not audio_file.lower().endswith(".wav"):
        converted = f"tmp_{unique_id}_conv.wav"
        try:
            subprocess.run(["ffmpeg", "-y", "-i", audio_file, converted], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            audio_file = converted
        except Exception as e:
            # cleanup original and return error
            try:
                os.remove(audio_file)
            except:
                pass
            return f"ERROR converting audio to wav: {str(e)}"

    # Transcribe with whisper
    try:
        # Use a modest model; change to "small" or "medium" for better accuracy if you have resources
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"ERROR transcribing with Whisper: {str(e)}"
    finally:
        # Cleanup audio files (original + converted)
        for f in glob.glob(f"tmp_{unique_id}*"):
            try:
                os.remove(f)
            except:
                pass

    return text

# -----------------------
# Main Processing Flow (wrapped in glass-card)
# -----------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📥 Ingest Sources")
st.markdown("<div class='small-muted'>Paste article URLs or YouTube links (one per line). The app will scrape/transcribe, chunk, embed and build a FAISS knowledge base.</div>", unsafe_allow_html=True)

if process_btn:
    if not urls.strip():
        st.warning("⚠️ Please enter at least one URL in the sidebar.")
    else:
        with st.spinner("✨ Fetching content and building knowledge base..."):
            url_list = [u.strip() for u in urls.split("\n") if u.strip()]
            documents = []

            for url in url_list:
                st.write(f"📥 Processing: {url}")

                try:
                    if "youtube.com" in url or "youtu.be" in url:
                        text = whisper_transcribe_youtube(url)
                    else:
                        text = load_url_with_playwright(url)
                except Exception as e:
                    text = f"ERROR processing {url}: {str(e)}"

                if isinstance(text, str) and text.startswith("ERROR"):
                    st.error(text)
                    continue

                if not text or len(text.strip()) < 30:
                    st.error(f"❌ Could not extract meaningful content from: {url}")
                    continue

                documents.append(Document(page_content=text, metadata={"source": url}))

            if not documents:
                st.error("❌ No valid content extracted. FAISS index cannot be created.")
            else:
                # chunking
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(documents)
                if not chunks:
                    st.error("❌ Document splitting returned no chunks.")
                else:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    db = FAISS.from_documents(chunks, embeddings)
                    db.save_local(DB_FAISS_PATH)
                    st.success("✅ Knowledge base created and saved to FAISS.")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Q&A card (glass)
# -----------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("💬 Ask Questions About the Data")
st.markdown("<div class='small-muted'>Type a question about the processed content. The model will search the FAISS index and answer using local Llama (via Ollama).</div>", unsafe_allow_html=True)

question = st.text_input("Enter your question:", placeholder="e.g., Summarize the key takeaways from the video or article.")
ask_btn = st.button("🔍 Analyze")

if ask_btn:
    if not os.path.exists(DB_FAISS_PATH):
        st.error("⚠️ FAISS index not found. Process sources first from the sidebar.")
    elif not question or not question.strip():
        st.warning("⚠️ Please type a question.")
    else:
        with st.spinner("🔎 Retrieving and generating answer..."):
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

                llm = Ollama(model="llama3", temperature=0.7)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=db.as_retriever(),
                    return_source_documents=True,
                )

                result = qa.invoke({"query": question})
                answer = result.get("result", "No answer returned.")
                sources = result.get("source_documents", [])

            except Exception as e:
                st.error(f"Error during retrieval/LLM: {str(e)}")
                answer = None
                sources = []

        if answer:
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            with st.expander("🔗 Source documents"):
                if sources:
                    for doc in sources:
                        src = doc.metadata.get("source", "Unknown")
                        st.markdown(f"- <span class='source-list'>{src}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='small-muted'>No sources returned.</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer card
# -----------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>"
            "<div><strong>Developed by Ishaan Farooq</strong><div class='small-muted'>Whisper • yt-dlp • Playwright • FAISS • Llama</div></div>"
            "<div class='small-muted'>Tip: Use normal YouTube links (https://www.youtube.com/watch?v=ID)</div>"
            "</div>",
            unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
