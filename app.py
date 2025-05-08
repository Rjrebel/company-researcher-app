from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

# Load environment variables from .env file
load_dotenv()


# Pass the API keys to the respective classes
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
    api_key=os.getenv("MISTRAL_API_KEY")
)


search = TavilySearchResults(max_results=2, api_key=os.getenv("TAVILY_API_KEY"))

tools = [search]

llm_with_tool = llm.bind_tools(tools)

agent_executor = create_react_agent(llm, tools)

prompt = """

You are a highly skilled Company Researcher Agent. Given the name of a company, your task is to research and provide a comprehensive, well-structured summary including the following details (as available):

Basic Information:

Company Name

Industry/Sector

Year Founded

Headquarters Location

Founders and Key Executives (CEO, CTO, etc.)

Number of Employees

Business Overview:

Description of products/services offered

Primary business model and revenue sources

Market position and notable partnerships

Financial Overview: (if public or available)

Latest revenue and profit figures

Funding rounds, investors, and valuation (if startup)

Stock performance (if publicly traded)

Recent News & Developments:

Acquisitions, product launches, major announcements

Legal or regulatory issues, if any

Competitors and Market Landscape:

Key competitors

Market share or positioning

Hiring and Culture:

Open roles or hiring trends

Company culture highlights (from Glassdoor, etc.)

Present the output in a clean, readable markdown format. If any information is not available, clearly indicate "Not Available" or "N/A".

"""


from langchain_core.messages import SystemMessage, HumanMessage


for step in agent_executor.stream(
    {"messages": [
        SystemMessage(content=prompt),
        HumanMessage(content="Tell me about TCS company?"),
        ]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()