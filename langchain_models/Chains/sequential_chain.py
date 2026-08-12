from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    provider="featherless-ai"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic' : 'Tell me about LLM'})

print(result)

