from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "chat_history.txt")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    provider="featherless-ai"
)

model = ChatHuggingFace(llm=llm)



# Prompt Template

chat_template = ChatPromptTemplate([
    (
        'system',
        '''You are a helpful customer support agent.

ONLY answer from the chat history provided.

If the answer is NOT in chat history, say:
"I could not find that information in our records."
'''
    ),

    MessagesPlaceholder(variable_name='chat_history'),

    ('human', '{query}')
])


# ===== LOAD CHAT HISTORY =====

chat_history = []


def load_history():

    chat_history.clear()

    try:

        with open(file_path, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                if line.startswith("Human:"):

                    content = line.replace("Human:", "", 1).strip()

                    if content:
                        chat_history.append(
                            HumanMessage(content=content)
                        )

                elif line.startswith("AI:"):

                    content = line.replace("AI:", "", 1).strip()

                    if content:
                        chat_history.append(
                            AIMessage(content=content)
                        )

    except FileNotFoundError:

        pass


# ===== SAVE HISTORY =====

def save_message(role, content):

    with open(file_path, "a") as f:

        f.write(f"{role}: {content}\n")


# ===== CLEAN TEXT =====

def get_text(response):

    if isinstance(response.content, str):

        return response.content

    elif isinstance(response.content, list):

        return " ".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    return str(response)


# Load old chat

load_history()


# ===== CHAT LOOP =====

print("🤖 Chatbot started (type 'exit' to stop)\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":

        print("goodbye")

        break


    # Reload full history

    load_history()


    # Create prompt

    prompt = chat_template.invoke({
        'chat_history': chat_history,
        'query': user_input
    })


    # LLM response

    response = model.invoke(prompt)

    ai_reply = get_text(response)

    print("AI:", ai_reply)


    # Save messages AFTER response

    save_message("Human", user_input)

    save_message("AI", ai_reply)