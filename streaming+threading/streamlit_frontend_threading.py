import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

#-----------------utility function------------------------------
def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id=generate_thread_id()
    st.session_state.thread_id=thread_id
    add_thread(st.session_state.thread_id,'New Chat')
    st.session_state.message_history=[]

def add_thread(thread_id,name='New Chat'):
    if thread_id not in st.session_state.chat_thread:
        st.session_state.chat_thread[thread_id]=name

def load_conversation(thread_id):
    return chatbot.get_state(config={"configurable":{"thread_id":thread_id}}).values.get('messages', [])

#----------------session setup-----------------------------------
if 'message_history' not in st.session_state:
    st.session_state.message_history=[]

if 'thread_id' not in st.session_state:
    st.session_state.thread_id=generate_thread_id()

if 'chat_thread' not in st.session_state:
    st.session_state.chat_thread={}
    add_thread(st.session_state.thread_id,"New Chat")

#-----------------sidebar ui-------------------------------------

st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header('My conversation')

for thread_id,name in reversed(list(st.session_state.chat_thread.items())):
    if st.sidebar.button(name,key=f"thread_{thread_id}"):
        st.session_state.thread_id=thread_id
        messages=load_conversation(thread_id)

        temp_messages=[]
        for message in messages:
            if isinstance(message,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role':role,'content':message.content})
        
        st.session_state.message_history = temp_messages



#-----------------showing all content----------------------------
for message in st.session_state.message_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input=st.chat_input('Type Here')

#-----------------user input and streaming-----------------------
if user_input:
    if st.session_state.chat_thread[st.session_state.thread_id] == "New Chat":
        st.session_state.chat_thread[st.session_state.thread_id] = user_input[:30]

    st.session_state.message_history.append({'role':'user',"content":user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    CONFIG={"configurable":{"thread_id":st.session_state['thread_id']}}

    ai_message=""
    with st.chat_message('assistant'):
        placeholder = st.empty()
        for message_chunk,metadata in chatbot.stream(
                {"messages":[HumanMessage(content=user_input)]},config=CONFIG,
                stream_mode='messages'
        ):
            ai_message+=message_chunk.content
            placeholder.markdown(ai_message)
        st.session_state.message_history.append({'role':'assistant',"content":ai_message})