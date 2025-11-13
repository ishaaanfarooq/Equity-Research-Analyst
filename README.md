# 🧠 Equity Research Analyst – AI-Powered Market Intelligence Tool

A modern, fully local **Equity Research & News Analysis AI** that can analyze:

### ✔ News articles  
### ✔ Financial reports  
### ✔ Market research  
### ✔ YouTube videos (ANY video — Whisper transcription)

Built using **LangChain**, **Ollama (Llama 3)**, **FAISS**, **Playwright**, **yt-dlp**, **Whisper**, and a stunning **Glassmorphism UI**.

---

## ✨ Features

### 🔍 **1. Multi-Source Content Extraction**
- Paste **any news article URL**
- Paste **any YouTube link** (supports: no captions, region-locked, Shorts, podcasts, aged videos)
- Auto-scraped using **Playwright** (for articles)
- Auto-transcribed using **Whisper + yt-dlp** (for YouTube)

### 🧠 **2. Local AI Processing**
- Uses **Llama3** via **Ollama** (no API key needed)
- Embeddings via **HuggingFace MiniLM**
- Fast local retrieval using **FAISS**

### 💬 **3. Ask Natural Questions**
After processing, ask:
- “Summarize the key insights.”
- “What risks does the article mention?”
- “Explain this video like I'm a beginner.”
- “What is the market outlook based on this content?”

### 🎨 **4. Modern Glassmorphism UI**
- Apple/Vercel style glass cards  
- Neon gradients  
- Smooth blur effects  
- Clean sidebar inputs  
- Professionally styled result layout  

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit + Custom CSS (Glassmorphism) |
| LLM | Llama 3 (via Ollama) |
| Embeddings | HuggingFace MiniLM |
| Vector DB | FAISS |
| Web Scraping | Playwright |
| Audio Transcription | Whisper + yt-dlp |
| Logic Framework | LangChain |

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ishaaanfarooq/Equity-Research-Analyst.git
cd Equity-Research-Analyst
