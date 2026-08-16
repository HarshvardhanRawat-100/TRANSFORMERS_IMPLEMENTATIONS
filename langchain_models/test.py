import langchain
print(langchain.__version__)

from dotenv import load_dotenv
import os
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv("/Users/harshvardhanrawat/Desktop/TRANSFORMER/.env")

print(os.getenv("GEMINI_API_KEY_2"))

