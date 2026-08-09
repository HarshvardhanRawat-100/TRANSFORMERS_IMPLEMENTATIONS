from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash",temperature=0.7)

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

# ===== CHAT LOOP =====
print("🤖 Chatbot started (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add user message to memory
    chat_history.append(HumanMessage(content=user_input))
    save_message("Human", user_input)

    # Create prompt with history
    prompt = chat_template.invoke({
        'chat_history': chat_history,
        'query': user_input
    })

    # Get response from LLM
    response = model.invoke(prompt)

    ai_reply = response.text
    print("AI:", ai_reply)

    # Save AI response
    chat_history.append(AIMessage(content=ai_reply))
    save_message("AI", ai_reply)