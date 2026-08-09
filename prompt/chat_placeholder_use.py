from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import re

# ✅ Load env
load_dotenv()

# ✅ Correct file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "chat_history.txt")

# ✅ Use correct Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7
)

# ✅ Prompt Template
chat_template = ChatPromptTemplate([
    ('system',
     '''You are a helpful customer support agent.

If user asks about an order:

- Valid order numbers are: #123, #456, #789
- If order number is NOT in this list, reply:
  "Your order is not in our list."
'''),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# ===== LOAD CHAT HISTORY =====
chat_history = []

def load_history():
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("Human:"):
                    chat_history.append(
                        HumanMessage(content=line.replace("Human:", "").strip())
                    )
                elif line.startswith("AI:"):
                    chat_history.append(
                        AIMessage(content=line.replace("AI:", "").strip())
                    )
    except FileNotFoundError:
        pass


# ===== SAVE CHAT HISTORY =====
def save_message(role, content):
    with open(file_path, "a") as f:
        f.write(f"{role}: {content}\n")


# Load previous chat
load_history()

# ===== CHAT LOOP =====
print("🤖 Chatbot started (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    # ✅ EXIT FIRST
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # ✅ Extract order from user input
    user_orders = re.findall(r"#\d+", user_input)

    # ✅ Valid orders (hardcoded)
    valid_orders = ["#123", "#456", "#789"]

    # ===== CUSTOM ORDER LOGIC =====
    if user_orders:
        if any(order in valid_orders for order in user_orders):
            ai_reply = f"Your order {user_orders[0]} is found and is being processed."
        else:
            ai_reply = "Your order is not in our list."
    else:
        # ===== LLM RESPONSE =====
        chat_history.append(HumanMessage(content=user_input))
        save_message("Human", user_input)

        prompt = chat_template.invoke({
            'chat_history': chat_history,
            'query': user_input
        })

        response = model.invoke(prompt)

        ai_reply = response.text

    # ✅ PRINT
    print("AI:", ai_reply)

    # ✅ SAVE AI RESPONSE
    chat_history.append(AIMessage(content=ai_reply))
    save_message("AI", ai_reply)