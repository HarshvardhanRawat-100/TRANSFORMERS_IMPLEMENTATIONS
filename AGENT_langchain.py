import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

load_dotenv()

# LLM
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY_2")
)

# Tool
search_tool = DuckDuckGoSearchRun()

# Agent
agent = create_agent(
    model=model,
    tools=[search_tool],
    system_prompt="""
    You are a helpful research agent.
    For every factual question, use the DuckDuckGo search tool
    to verify the information before answering.
    """
)

# Run agent
response = agent.invoke({
    "messages": [
        (
            "user",
            "Find the capital of Madhya Pradesh and the top 5 places in Madhya Pradesh."
        )
    ]
})

# Print only the final answer
print(response["messages"][-1].content)