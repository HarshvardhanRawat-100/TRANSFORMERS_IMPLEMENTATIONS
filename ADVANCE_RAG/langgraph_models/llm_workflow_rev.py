from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY_4"))


