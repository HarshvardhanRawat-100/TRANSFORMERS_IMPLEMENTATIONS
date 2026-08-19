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


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the currency conversion factor between two currencies."""

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    data = response.json()

    return data["conversion_rate"]


@tool
def convert(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Converts the base currency value using the conversion rate."""

    return base_currency_value * conversion_rate


llm_with_tools = model.bind_tools([
    get_conversion_factor,
    convert
])


messages = [
    HumanMessage(
        content="What is the conversion factor between INR and USD, "
                "and convert 10 INR to USD."
    )
]


# 1. LLM decides which tool it needs
ai_message = llm_with_tools.invoke(messages)

print("AI TOOL CALLS:")
print(ai_message.tool_calls)

messages.append(ai_message)


# 2. Execute get_conversion_factor
conversion_rate = None

for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "get_conversion_factor":

        tool_message = get_conversion_factor.invoke(tool_call)

        conversion_rate = float(tool_message.content)

        print("Conversion rate:", conversion_rate)

        messages.append(tool_message)


# 3. Execute convert manually
converted_value = convert.invoke({
    "base_currency_value": 10,
    "conversion_rate": conversion_rate
})

print("Converted value:", converted_value)


# 4. Give final result to LLM
messages.append(
    HumanMessage(
        content=f"The conversion rate is {conversion_rate}. "
                f"10 INR is {converted_value} USD. "
                f"Give the user the final answer."
    )
)

# 5. Final LLM response
final_response = llm_with_tools.invoke(messages)
print("\nFINAL ANSWER:")
print(final_response.content)