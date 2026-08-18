import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash",temperature=0.7 , google_api_key=os.getenv("GOOGLE_API_KEY_2"))