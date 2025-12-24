import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("human", "{question}")
    ]
)

def generate_response(question, model, temperature):
    llm = Ollama(
        model=model,
        temperature=temperature
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# App title
st.title("Enhanced Q&A Chatbot with Ollama 🦙")

# Sidebar
st.sidebar.title("Settings")

model = st.sidebar.selectbox(
    "Select an Ollama model",
    [
        "llama2",
        "gemma:2b"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

# Main UI
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    with st.spinner("Thinking..."):
        response = generate_response(user_input, model, temperature)
    st.write("**Assistant:**")
    st.write(response)
else:
    st.write("Ask any query to get started.")
