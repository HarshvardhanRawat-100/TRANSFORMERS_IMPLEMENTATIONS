from langgraph.graph import StateGraph, START, MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
True
model = ChatGoogleGenerativeAI
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

graph = builder.compile()
