# 💼 HR-RAG Assistant — NexaTech Indonesia

A simple Retrieval-Augmented Generation (RAG) based HR chatbot built with Streamlit, ChromaDB, LangChain, HuggingFace Embeddings, and DeepSeek API.

This project was created as an experimental MVP to simulate an internal HR knowledge assistant for **PT NexaTech Indonesia (fictional company)**.

It helps employees quickly retrieve HR-related information such as onboarding, leave policies, payroll, benefits, and company rules through conversational search.

You can see it in https://hr-rag.streamlit.app/

---

## 📌 Project Overview

This is my **first attempt at building a simple RAG-based chatbot**.

The goal of this project was to understand how a lightweight Retrieval-Augmented Generation pipeline works end-to-end:

- Store internal HR knowledge
- Convert documents into embeddings
- Save vectors in a vector database
- Retrieve relevant context
- Generate contextual responses using an LLM

This is still an early MVP, and there are many areas that can be improved.

---

## 🚀 Features

- HR chatbot for internal company knowledge
- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- Vector storage using ChromaDB
- Context-aware response generation
- Streamlit chat interface
- Local vector database (no Pinecone)
- Lightweight and beginner-friendly architecture

---

## 🧠 Use Cases

Employees can ask questions like:

- How many annual leave days do I get?
- What documents are required for onboarding?
- How does reimbursement work?
- What is the payroll schedule?
- What happens during resignation?

---

## ⚙️ Tech Stack

- **Frontend / UI** → Streamlit
- **Framework** → LangChain
- **LLM API** → DeepSeek API
- **Embeddings** → HuggingFace Sentence Transformers
- **Vector Database** → ChromaDB
- **Language** → Python
- **Environment Handling** → python-dotenv

---

## 🏗️ Architecture

```txt
HR Documents (.md)
        ↓
Text Embedding
(HuggingFace)
        ↓
Vector Storage
(ChromaDB)
        ↓
Retriever
        ↓
Relevant Context
        ↓
DeepSeek LLM
        ↓
Response in Streamlit UI
