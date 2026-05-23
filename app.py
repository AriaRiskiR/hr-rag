import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DOCS_DIR = "docs"
DB_DIR = "chroma_db"

st.set_page_config(page_title="HR Chatbot")
st.title("💼 HR Assistant")

# -------------------------
# Vector DB
# -------------------------
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(DB_DIR):
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

    docs = []

    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".md"):
            path = os.path.join(DOCS_DIR, filename)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename}
                    )
                )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    return vector_db


vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# -------------------------
# LLM
# -------------------------
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    st.error("DEEPSEEK_API_KEY belum ditemukan. Cek file .env kamu.")
    st.stop()

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.2,
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Kamu adalah HR chatbot internal perusahaan NexaTech Indonesia.

Aturan:
- Jawab hanya berdasarkan konteks dokumen HR.
- Jika tidak ada info, bilang tidak ditemukan.
- Bahasa Indonesia.
- Singkat, jelas, profesional.
- Kalau user tanya onboarding, leave, payroll, benefits, jawab seperti HR assistant.

Context:
{context}
"""),
    ("human", "{question}")
])

# -------------------------
# Chat history
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Tanya HR policy...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(
        [doc.page_content for doc in relevant_docs]
    )

    chain = prompt | llm
    answer = chain.invoke({
        "context": context,
        "question": question
    }).content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)