from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langsmith import traceable
import operator

load_dotenv()
os.environ['LANGCHAIN_PROJECT']="langgraph-parent-subgraph-monitoring"

class SubState(TypedDict):

    input_text: str
    translated_text: str


subgraph_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
                                      google_api_key=os.getenv("GOOGLE_API_KEY_4"))

@traceable(
    name="Victorian English Translator",
    tags=["translation", "subgraph"],
    metadata={"language": "Victorian English"}
)
def translate_text(state: SubState):

    prompt = f"""
Translate the following text to Victorian English.
Keep it natural and clear. Do not add extra content.

Text:
{state["input_text"]}
""".strip()
    
    translated_text = subgraph_llm.invoke(prompt).text

    return {'translated_text': translated_text}


subgraph_builder = StateGraph(SubState)

subgraph_builder.add_node('translate_text', translate_text)

subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

subgraph = subgraph_builder.compile()

class ParentState(TypedDict):

    question: str
    answer_eng: str
    answer_vic_eng: str

parent_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
                                      google_api_key=os.getenv("GOOGLE_API_KEY_4"))


@traceable(
    name="Generate English Answer",
    tags=["generation", "parent-graph"]
)
def generate_answer(state: ParentState):

    answer = parent_llm.invoke(f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}").text
    return {'answer_eng': answer}

@traceable(
    name="Call Translation Subgraph",
    tags=["subgraph-invocation"]
)

def translate_answer(state: ParentState):

    # call the subgraph
    result = subgraph.invoke({'input_text': state['answer_eng']})

    return {'answer_vic_eng': result['translated_text']}
parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", translate_answer)

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)

graph = parent_builder.compile()

result = graph.invoke(
    {"question": "explain what is the best teaching from batman which christopher nolam made"},
    config={
        "run_name": "BATMAN",
        "tags": [
            "langgraph",
            "parent-subgraph",
            "gemini",
            "translation"
        ],
        "metadata": {
            "model": "gemini-3.5-flash",
            "task": "question-answering-and-translation",
            "version": "1.0"
        }
    }
)

print("\nENGLISH ANSWER:\n", result["answer_eng"])
print("\nVICTORIAN ENGLISH:\n", result["answer_vic_eng"])