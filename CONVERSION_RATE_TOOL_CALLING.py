import requests
import os
from dotenv import load_dotenv
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY_2")
)

API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")


# ---------------- TOOL 1 ----------------

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetches the currency conversion factor between two currencies.
    """

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    data = response.json()

    return data["conversion_rate"]


# ---------------- TOOL 2 ----------------

@tool
def convert(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    Converts the base currency value using the conversion rate.
    """

    return base_currency_value * conversion_rate


# ---------------- TOOL BINDING ----------------

llm_with_tools = model.bind_tools([
    get_conversion_factor,
    convert
])


# ---------------- USER MESSAGE ----------------

messages = [
    HumanMessage(
        content="What is the conversion factor between INR and USD, "
                "and convert 10 INR to USD."
    )
]


# ---------------- FIRST LLM CALL ----------------

ai_message = llm_with_tools.invoke(messages)

print("AI TOOL CALLS:")
print(ai_message.tool_calls)

messages.append(ai_message)


# ---------------- TOOL EXECUTION ----------------

conversion_rate = None

for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "get_conversion_factor":

        tool_message1 = get_conversion_factor.invoke(tool_call)

        conversion_rate = tool_message1.content

        messages.append(tool_message1)

    elif tool_call["name"] == "convert":

        tool_call["args"]["conversion_rate"] = conversion_rate

        tool_message2 = convert.invoke(tool_call)

        messages.append(tool_message2)


# ---------------- FINAL LLM CALL ----------------

final_response = llm_with_tools.invoke(messages)

print("\nFINAL ANSWER:")
print(final_response.content)