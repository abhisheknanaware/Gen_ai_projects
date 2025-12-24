import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with OpenAI"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to user queriers"),
        ("human","{question}")
    ]
)

def generate_response(question,api_key,model,temperature,max_tokens):
    openai.api_key=api_key
    llm=ChatOpenAI(model=model,temperature=temperature,max_tokens=max_tokens)
    chain=prompt|llm|StrOutputParser()
    response=chain.invoke({"question":question})
    return response


#title of the app
st.title("Enhanced Q&A Chatbot With OpenAi")

#sidebar
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API key:",type="password")

##Drop down to select Various Open Ai model
llm=st.sidebar.selectbox("Select an Open AI model",["gpt-4o", "gpt-4o-mini"])

#adjust response paramter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max toekns",min_value=50,max_value=300,value=150)

#main Interface For user input
st.write("Go head And Ask Any questions")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
elif user_input:
    st.write("plz enter api key")
else:
    st.write("Ask Any Query")


