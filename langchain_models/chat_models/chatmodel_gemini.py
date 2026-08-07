from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash",temperature=1.7)
result = model.invoke("give 3 best quotes from batman movie universe which nolan made")
print(result.text)