import streamlit as st
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

st.title("Conversational RAG with PDF")

api_key = st.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Please enter Groq API key")
    st.stop()

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)
session_id=st.text_input("Session Id",value="default_session")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Session store
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the context to answer. If unknown, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    question = st.text_input("Ask a question")
    session_history=get_session_history(session_id)

    if question:
        response = conversational_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        st.success(response)
        st.subheader("Chat History")

        for msg in session_history.messages:
            if msg.type == "human":
                st.markdown(f"**You:** {msg.content}")
            else:
                st.markdown(f"**Assistant:** {msg.content}")
