from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os 

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system' , 'you are a helpfule {domain} expert') ,
    ('human', 'explain in simple term what is {topics}')
])

prompt = chat_template.invoke({'domain' : 'cricket' , 'topics' : 'spin'})

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    provider="featherless-ai"
) 

model = ChatHuggingFace(llm = llm)

response = model.invoke(prompt)
print(response.content)