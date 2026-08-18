import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os 

load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash",temperature=0.7 , google_api_key=os.getenv("GOOGLE_API_KEY_2"))

# tool create

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b
 
# tool binding 
llm_with_tools = model.bind_tools([multiply]) 
 
query = HumanMessage('multiply 3 with 1000')

message = [query]

# tool calling 

result = llm_with_tools.invoke(message)

message.append(result)

# tool execution

tool_result = multiply.invoke(result.tool_calls[0])
message.append(tool_result)

# printing result 

print(llm_with_tools.invoke(message).content)