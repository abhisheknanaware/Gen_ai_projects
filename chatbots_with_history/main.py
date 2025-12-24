import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##set up streamlit app
st.title("Conversational RAG with Pdf uploads and chat history")
st.write("Upload Pdf's and Chat with their content")

##input the groq api key 
api_key=st.text_input("Enter your Groq Api key:",type="password")

##check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="meta-llama/llama-4-scout-17b-16e-instruct")
    session_id=st.text_input("Session Id",value="default_session")

    ##statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_file=st.file_uploader("Choose a file type",type="pdf")
    #process uploaded files
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
            
        loader=PyPDFLoader("temp.pdf")
        documents=loader.load()
    
    #split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        ##new prompt
        contextualize_q_system_prompt=(
        "Given a chat history and latest user question"
        "which might reference context in chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history,Do not answer question,"
        "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt=ChatPromptTemplate.from_messages([
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{question}"),
        ])

        answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. "
                "Use the following context to answer the question. "
                "If you don't know, say you don't know. "
                "Use three sentences maximum.\n\n{context}"
            ),
            ("human", "{question}"),
        ]
        )

        question_rewriter = (
            contextualize_q_prompt
            | llm
            | StrOutputParser()
        )

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        history_aware_retriever = RunnableLambda(
            lambda x: format_docs(
                retriever.invoke(
                    question_rewriter.invoke(
                        {
                            "question": x["question"],
                            "chat_history": x["chat_history"],
                        }
                    )
                )
            )
        )
        answer_chain = (
            answer_prompt
            | llm
            | StrOutputParser()
        )
        rag_chain = (
            {
                "context": history_aware_retriever,
                "question": lambda x: x["question"],
            }
            | answer_chain
        )

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        convernsational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        user_input=st.text_input("Ask your question")
        if user_input:
            session_history=get_session_history(session_id)
            response=convernsational_rag_chain.invoke(
                {"question":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.success(response)
            st.write("cht history:",session_history.messages)
else:
    st.warning("please enter the Groq api key")