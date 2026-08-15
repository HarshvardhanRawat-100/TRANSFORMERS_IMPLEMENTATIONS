from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough

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

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_gen_chain , parallel_chain)
print(chain.invoke({'topic' : 'AI'}))
chain.get_graph().print_ascii()