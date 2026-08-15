from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()
parser = StrOutputParser()
# Define the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    provider="featherless-ai"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Write joke about this {topic}" ,
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = "Write summary of this {text}" ,
    input_variables = ['text']
)
chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)
print(chain.invoke({'topic' : 'AI'}))
chain.get_graph().print_ascii()