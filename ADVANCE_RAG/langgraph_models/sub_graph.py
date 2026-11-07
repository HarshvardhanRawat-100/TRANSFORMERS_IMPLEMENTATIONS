from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['LANGCHAIN_PROJECT']='langraph monitoring using langsmith'

class SubState(TypedDict):

    input_text: str
    translated_text: str


subgraph_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
                                      google_api_key=os.getenv("GOOGLE_API_KEY_4"))

def translate_text(state: SubState):

    prompt = f"""
Translate the following text to Victorian English.
Keep it natural and clear. Do not add extra content.

Text:
{state["input_text"]}
""".strip()
    
    translated_text = subgraph_llm.invoke(prompt).content

    return {'translated_text': translated_text}


subgraph_builder = StateGraph(SubState)

subgraph_builder.add_node('translate_text', translate_text)

subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

subgraph = subgraph_builder.compile()

class ParentState(TypedDict):

    question: str
    answer_eng: str
    answer_hin: str

    