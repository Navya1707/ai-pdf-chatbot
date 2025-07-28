import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from typing import List, Tuple

# Load API key from Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Tuple[str, str]] = []

def add_to_chat_history(question: str, answer: str):
    st.session_state.chat_history.append((question, answer))

def display_chat_history():
    st.markdown("### 🕑 Previous Q&A")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {q}"):
            st.markdown(f"**A:** {a}")

# -------------------- PDF Processing --------------------
def extract_text_from_pdfs(uploaded_files):
    """Extract raw text from uploaded PDF files."""
    combined_text = ""
    for file in uploaded_files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            combined_text += page.extract_text()
    return combined_text

def split_text_into_chunks(text, chunk_size=1500, overlap=200):
    """Split text into chunks with overlap for better context."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# -------------------- Embedding & Vector Store --------------------
def store_text_embeddings(chunks, index_path="faiss_index"):
    """Generate and save embeddings to FAISS vector store."""
    if not chunks:
        raise ValueError("No text chunks found for embedding.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    vector_db.save_local(index_path)

def load_vector_store(index_path="faiss_index"):
    """Load the existing FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# -------------------- QA Chain Setup --------------------
def create_qa_chain():
    """Build a Gemini-powered QA chain using a custom prompt."""
    template = """
    You are a helpful assistant. Provide accurate, detailed answers based only on the given context.
    If unsure, respond: "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# -------------------- Response Handling --------------------
def generate_answer(question, reference_idx=None):
    """Search and answer using vector store, optionally with reference to past Q&A."""
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(question)
    
    # Use selected previous Q&A for cross-reference
    reference_context = ""
    if reference_idx is not None and 0 <= reference_idx < len(st.session_state.chat_history):
        prev_q, prev_a = st.session_state.chat_history[reference_idx]
        reference_context = f"\nPrevious Question: {prev_q}\nPrevious Answer: {prev_a}\n"

    # Append reference context to current docs
    for doc in docs:
        doc.page_content = reference_context + doc.page_content

    chain = create_qa_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    
    st.write("💬 **Answer:**", result["output_text"])
    add_to_chat_history(question, result["output_text"])


# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config(page_title="PDF Chat Agent", page_icon="📄", layout="wide")
    st.title("📚 Ask Anything from Your PDFs")
    st.markdown("Chat with your uploaded documents here!")

    init_chat_history()

    with st.sidebar:
        st.markdown("---")
        st.subheader("📂 Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("📥 Process PDFs"):
            with st.spinner("Processing documents..."):
                text = extract_text_from_pdfs(uploaded_files)
                chunks = split_text_into_chunks(text)
                store_text_embeddings(chunks)
                st.success("PDFs processed!")

        st.markdown("---")
        if st.session_state.chat_history:
            st.subheader("🔁 Cross-Reference")
            reference_options = [f"Q{i+1}: {q}" for i, (q, _) in enumerate(st.session_state.chat_history)]
            selected_ref = st.selectbox("Refer to a previous question", ["None"] + reference_options)
            reference_idx = reference_options.index(selected_ref) if selected_ref != "None" else None
        else:
            reference_idx = None

        st.caption("Created by Navya • Powered by LangChain + Gemini")

    st.markdown("### 🔍 Ask a Question")
    user_query = st.text_input("Type your question here...")

    if user_query:
        generate_answer(user_query, reference_idx)

    if st.session_state.chat_history:
        display_chat_history()


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
