from langgraph.graph import StateGraph, START, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

import os 

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY_4"))

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

graph = builder.compile()

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}
graph.invoke({"messages": [{"role": "user", "content": " My name is BATMAN."}]}, config)

print(graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config2))