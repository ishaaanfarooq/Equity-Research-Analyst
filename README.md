# 🧠 Equity Research Analyst – News & Market Intelligence Tool

Analyze stock market news and financial reports directly from URLs.  
Built with **LangChain**, **Llama (via Ollama)**, and **FAISS** for efficient retrieval-based Q&A.

---

## 🚀 Features
- 🔗 Load news or financial articles via URLs
- ⚙️ Automatically extract and process text
- 🧬 Generate vector embeddings using HuggingFace
- ⚡ Efficient retrieval using FAISS
- 💬 Ask natural language questions and get summarized insights
- 💾 Local FAISS storage for fast repeated analysis

---

## 🧩 Tech Stack
- **LangChain**
- **FAISS**
- **Ollama (Llama models)**
- **Streamlit**
- **Python 3.10+**

---

## 🧰 Installation

```bash
# Clone repository
git clone https://github.com/ishaaanfarooq/Equity-Research-Analyst.git
cd Equity-Research-Analyst

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # For Ubuntu/Mac
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt
