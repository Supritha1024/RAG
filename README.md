# 📄 Document Question Answering System using RAG

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) based Document Question Answering system using LangChain and Streamlit.

Users can upload a PDF document (Paracetamol document in this project) and ask questions about its content. The system retrieves the most relevant document chunks using vector similarity search and generates accurate answers using a language model.

---

## 🧠 Architecture

User Uploads PDF  
→ Document Chunking  
→ Embedding Generation (all-MiniLM-L6-v2)  
→ FAISS Vector Database  
→ Top-3 Relevant Chunk Retrieval  
→ LLM Answer Generation  
→ Display Answer + Source Chunks in Streamlit  

---

## 🛠️ Tech Stack

- Python  
- LangChain  
- HuggingFace Embeddings  
- FAISS  
- Streamlit  
- Transformers  
- PyPDF  

---

## 📦 Features

- Upload PDF document  
- Recursive document chunking  
- Semantic embeddings creation  
- FAISS vector storage  
- Top-3 relevant chunk retrieval  
- LLM-based answer generation  
- Display source documents  

---

## 🔍 Chunking Strategy

We used `RecursiveCharacterTextSplitter` with:

- chunk_size = 500  
- chunk_overlap = 100  

This ensures sufficient context per chunk while maintaining semantic continuity between chunks for better retrieval accuracy.

---

## ▶️ How to Run

### 1️⃣ Create Environment

```bash
conda create -n rag_env python=3.10
conda activate rag_env
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🎯 Learning Outcomes

- Document chunking techniques  
- Embedding generation  
- Vector similarity search  
- Retrieval pipelines  
- LLM integration  
- End-to-end RAG system implementation  

---

## 👩‍💻 Author

Mamilla Supritha  
B.Tech – AI & Data Science  