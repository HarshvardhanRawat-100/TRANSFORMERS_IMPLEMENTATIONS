from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    provider="featherless-ai"
) 

model = ChatHuggingFace(llm = llm)

# Prompt Template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# ===== LOAD CHAT HISTORY =====
chat_history = []

def load_history():
    try:
        with open("chat_history.txt", "r") as f:
            for line in f:
                if line.startswith("Human:"):
                    chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
                elif line.startswith("AI:"):
                    chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))
    except FileNotFoundError:
        pass

# ===== SAVE CHAT HISTORY =====
def save_message(role, content):
    with open("chat_history.txt", "a") as f:
        f.write(f"{role}: {content}\n")

# Load old chat
load_history()
