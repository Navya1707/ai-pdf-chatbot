# 📄 AI PDF Chatbot

An AI-powered chatbot that lets you upload PDF documents and query them in natural language.  
It extracts text, embeds the content, and uses a Large Language Model (LLM) to answer your questions — making it easier to interact with long or complex documents.

---

## 🚀 Features
- Upload one or more PDF files  
- Ask natural language questions about the content  
- Retrieves relevant sections before answering  
- Built with Streamlit for an interactive UI  

---

## 🛠️ Tech Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/)  
- **Backend**: Python (`chat_app.py`)  
- **LLM**: OpenAI API (configurable via secrets)  
- **Dependencies**: Listed in `requirements.txt`  

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Navya1707/ai-pdf-chatbot.git
cd ai-pdf-chatbot
2. Install dependencies
pip install -r requirements.txt
3. Add your API key
Create a .streamlit/secrets.toml file (or update the existing one):
OPENAI_API_KEY = "your_openai_api_key"
🔑 You’ll need an OpenAI key (or another supported LLM provider).
▶️ Running the App
Run the Streamlit app locally:
streamlit run chat_app.py
Then, open the local URL (usually http://localhost:8501) in your browser.
