from langchain_core.messages import SystemMessage , HumanMessage ,AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os 

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash",temperature=0.7)

chat_history = [
     SystemMessage(content='You are a helpful AI assistant')
]

while True :
    user_input = input("you : ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit' :
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content[0]["text"]))
    print("AI:", result.content[0]["text"])

print(chat_history)
 


  
                        