import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_community.callbacks import StreamlitCallbackHandler
import os
from langchain_core.prompts import ChatPromptTemplate

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can use tools."),
    ("placeholder", "{messages}")
])

api_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_wiki = WikipediaQueryRun(api_wrapper=api_wiki)

api_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_arxiv = ArxivQueryRun(api_wrapper=api_arxiv)

search=DuckDuckGoSearchRun(name="Search")
tools=[search,tool_wiki,tool_arxiv]
st.title("Langchain-chat with search")

#sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter the OpenAI API key", type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,I'm a chatbot who can search the web.How can I help you?"}
    ]

for msg in st.session_state.messages:
     st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="Ask anything..."):
     st.session_state.messages.append({"role":"user","content":prompt})
     st.chat_message("user").write(prompt)

     if not api_key:
        st.error("Please enter your openAi API key.")
        st.stop()

     llm=ChatOpenAI(api_key=api_key,model="gpt-4o-mini",streaming=True)


     agent=create_react_agent(
          model=llm,
          tools=tools,
          prompt=agent_prompt
        )
     
     with st.chat_message("assistant"):
          st_cb = StreamlitCallbackHandler(st.container())
          response = agent.invoke(
            {"messages": st.session_state.messages},
            config={"callbacks": [st_cb]}
            )

          final_answer = response["messages"][-1].content
          st.session_state.messages.append(
            {"role": "assistant", "content": final_answer}
          )

          st.write(final_answer)





