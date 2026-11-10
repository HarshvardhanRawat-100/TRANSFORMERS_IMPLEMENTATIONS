from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash",temperature = 0.9,
    google_api_key=os.getenv("GOOGLE_API_KEY_4"))

# Node
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
# Build graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Run ONCE (creates tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # Thread 1 (remembers)
    t1 = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is BATMAN"}]}, t1)
    out1 = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, t1)
    print("Thread-1:", out1["messages"][-1].content[0]["text"])