
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langsmith import traceable
import os

# Load environment variables
load_dotenv()

# LangSmith tracing using LANGCHAIN variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-parent-subgraph-monitoring"

# Verify configuration
print("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("API key loaded:", bool(os.getenv("LANGCHAIN_API_KEY")))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))

if not os.getenv("LANGCHAIN_API_KEY"):
    raise ValueError("LANGCHAIN_API_KEY missing in .env")

if not os.getenv("GOOGLE_API_KEY_4"):
    raise ValueError("GOOGLE_API_KEY_4 missing in .env")

# -------------------------------
# SUBGRAPH
# -------------------------------

class SubState(TypedDict):
    input_text: str
    translated_text: str


subgraph_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY_4")
)


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

    return {"translated_text": translated_text}


subgraph_builder = StateGraph(SubState)

subgraph_builder.add_node("translate_text", translate_text)

subgraph_builder.add_edge(START, "translate_text")
subgraph_builder.add_edge("translate_text", END)

subgraph = subgraph_builder.compile()


# -------------------------------
# PARENT GRAPH
# -------------------------------

class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_vic_eng: str


parent_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY_4")
)


@traceable(
    name="Generate English Answer",
    tags=["generation", "parent-graph"]
)
def generate_answer(state: ParentState):

    answer = parent_llm.invoke(
        f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}"
    ).text

    return {"answer_eng": answer}


@traceable(
    name="Call Translation Subgraph",
    tags=["subgraph-invocation"]
)
def translate_answer(state: ParentState):

    result = subgraph.invoke({
        "input_text": state["answer_eng"]
    })

    return {"answer_vic_eng": result["translated_text"]}


# Build parent graph
parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", translate_answer)

parent_builder.add_edge(START, "answer")
parent_builder.add_edge("answer", "translate")
parent_builder.add_edge("translate", END)

graph = parent_builder.compile()


# -------------------------------
# EXECUTION
# -------------------------------

result = graph.invoke(
    {
        "question": "Explain the best teaching from Batman, directed by Christopher Nolan."
    },
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
