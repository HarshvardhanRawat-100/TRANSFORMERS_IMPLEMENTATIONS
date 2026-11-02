
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os 

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY_4"))
