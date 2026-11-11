import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langgraph.store.memory import InMemoryStore

load_dotenv()

# creating a store
store = InMemoryStore()
# creating a namespace
namespace = ("user", "u1")

embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')
store = InMemoryStore(index={'embed': embedding_model, 'dims':1536})

store.put(namespace, "1", {"data": "User prefers concise answers over long explanations"})
store.put(namespace, "2", {"data": "User likes examples in Python"})
store.put(namespace, "3", {"data": "User usually works late at night"})
store.put(namespace, "4", {"data": "User prefers dark mode in applications"})
store.put(namespace, "5", {"data": "User is learning machine learning"})
store.put(namespace, "6", {"data": "User dislikes overly theoretical explanations"})
store.put(namespace, "7", {"data": "User prefers step-by-step reasoning"})
store.put(namespace, "8", {"data": "User is based in India"})
store.put(namespace, "9", {"data": "User likes real-world analogies"})
store.put(namespace, "10", {"data": "User prefers bullet points over paragraphs"})


items = store.search(namespace, query="what is the user currently learning", limit=1)

for item in items:
    print(item.value)
