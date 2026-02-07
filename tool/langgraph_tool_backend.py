from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import sqlite3

load_dotenv()
import os
#--------------llm----------------------
llm = ChatOpenAI()

#-------------tools--------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, subtract, multiply, divide
    """
    try:
        if operation == "add":
            res = first_num + second_num
        elif operation == "subtract":
            res = first_num - second_num
        elif operation == "multiply":
            res = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            res = first_num / second_num
        else:
            return {"error": f"Unsupported operation: {operation}"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": res
        }

    except Exception as e:
        return {"error": str(e)}
    
@tool
def get_stock_price(symbol:str)->dict:
    """
    fetch latest stock price for a given stock symbol (e.g., AAPL, GOOGL).using Alpaca vantage with api key in the url
    """
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=PAVRSENNZMP0T3DN"
    r=requests.get(url)
    return r.json()

@tool
def get_conversion_factor(base_currecy:str,target_currency:str)->dict:
    """This function fetches the currency conversion factor between a given base currency and a target currency"""

    url=f'https://v6.exchangerate-api.com/v6/d21877e2bfcc8e6a2bc53ff4/pair/{base_currecy}/{target_currency}'

    response=requests.get(url)
    res=response.json()
    return {"conversion_rate":res['conversion_rate']}

@tool
def currency_converter(base_currency_value:int,conversion_rate:float)->dict:
    "given a currency conversion rate this function calculate the target currency value from a given base currency value"
    output=base_currency_value*conversion_rate
    return {"converted_value":output}


tools=[search_tool,calculator,get_stock_price,get_conversion_factor,currency_converter]
# llm_with_tools=llm.bind_tools(tools)
#--------------chatbot state----------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

tool_node=ToolNode(tools)
#--------------graph and checkpointer----------------------

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

#---------------graph----------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
       all_threads.add(checkpoint.config['configurable']['thread_id']) 
    return list(all_threads)
