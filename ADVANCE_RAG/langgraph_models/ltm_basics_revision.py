import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langgraph.store.memory import InMemoryStore

load_dotenv()

# creating a store
store = InMemoryStore()
# creating a namespace
namespace = ("user", "u1")

