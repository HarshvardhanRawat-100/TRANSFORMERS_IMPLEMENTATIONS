import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash",temperature=0.7 , google_api_key=os.getenv("GOOGLE_API_KEY_2"))

search_tool = DuckDuckGoSearchRun()
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt
