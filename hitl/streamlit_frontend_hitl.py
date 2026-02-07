import streamlit as st
from langgraph_hitl_backend import chatbot,retrieve_all_threads
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid

#-----------------utility function------------------------------
def generate_thread_id():
    thread_id=uuid.uuid4()
    return str(thread_id)

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
    st.session_state.chat_thread = {
        thread_id: "Old Chat"
        for thread_id in retrieve_all_threads()
    }

    
add_thread(st.session_state.thread_id,"New Chat")

#-----------------sidebar ui-------------------------------------

st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header('My conversation')

for thread_id,name in reversed(list(st.session_state.chat_thread.items())):
    if st.sidebar.button(name,key=f"thread_{str(thread_id)}"):
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
    if st.session_state.chat_thread[st.session_state.thread_id] in ("New Chat", "Old Chat"):
        st.session_state.chat_thread[st.session_state.thread_id] = user_input[:30]


    st.session_state.message_history.append({'role':'user',"content":user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    CONFIG={"configurable":
            {"thread_id":st.session_state['thread_id']},
            "metadata":{
                "thread_id":st.session_state['thread_id']
            },
            "run_name":"chat_turn"
            }

    ai_message = ""

    with st.chat_message("assistant"):
        tool_status = None
        current_tool = None
        placeholder = st.empty()

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")

                #New tool started
                if current_tool != tool_name:
                    current_tool = tool_name

                    if tool_status is None:
                        tool_status = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )
                    else:
                        tool_status.update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True
                        )

            # 🤖 AI streaming
            elif isinstance(message_chunk, AIMessage):
                # Close tool status when AI starts responding
                if tool_status is not None:
                    tool_status.update(
                        label="✅ Tool finished",
                        state="complete",
                        expanded=False
                    )
                    tool_status = None
                    current_tool=None

                ai_message += message_chunk.content
                placeholder.markdown(ai_message)

    st.session_state.message_history.append({
        "role": "assistant",
        "content": ai_message
    })
