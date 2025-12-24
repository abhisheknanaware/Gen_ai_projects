import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()
##load the Groq Api key
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.environ["GROQ_API_KEY"]

llm=ChatGroq(groq_api_key=groq_api_key,model="meta-llama/llama-4-scout-17b-16e-instruct")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    please Provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{question}
    """
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        documents=st.session_state.loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=300)
        final_docs=text_splitter.split_documents(documents)
        st.session_state.vectors=FAISS.from_documents(final_docs,st.session_state.embeddings)

st.title("📄 Research Paper Q&A (RAG)")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    with st.spinner("Creating vector database..."):
        create_vector_embedding()
    st.success("Vector database is ready")

import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create document embeddings first.")
        st.stop()
    
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})

    rag_chain=(
        {"context":retriever| RunnableLambda(format_docs),
         "question":RunnablePassthrough()
         }
        |prompt
        |llm
        )

    start=time.process_time()
    response=rag_chain.invoke(user_prompt)
    elapsed = time.process_time() - start
    
     # Answer
    st.subheader("📌 Answer")
    st.write(response.content)
    st.caption(f"⏱ Response time: {elapsed:.2f} seconds")

    # Retrieved documents
    docs = retriever.invoke(user_prompt)

    with st.expander("📚 Document Similarity Search"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**Document {i}**")
            st.write(doc.page_content)
            st.write("-----")

