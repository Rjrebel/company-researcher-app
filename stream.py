# app.py

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM and tools
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
    api_key=os.getenv("MISTRAL_API_KEY")
)

search = TavilySearchResults(max_results=2, api_key=os.getenv("TAVILY_API_KEY"))
tools = [search]
agent_executor = create_react_agent(llm, tools)

# Streamlit UI
st.set_page_config(page_title="Company Researcher", layout="centered")
st.title("🔍 Company Researcher Agent")

company_name = st.text_input("Enter a Company Name", placeholder="e.g., TCS")

submit_button = st.button("Research Company")

# Prompt Template
system_prompt = """
You are a highly skilled Company Researcher Agent. Given the name of a company, your task is to research and provide a comprehensive, well-structured summary including the following details (as available):

Basic Information:
- Company Name
- Industry/Sector
- Year Founded
- Headquarters Location
- Founders and Key Executives (CEO, CTO, etc.)
- Number of Employees

Business Overview:
- Description of products/services offered
- Primary business model and revenue sources
- Market position and notable partnerships

Financial Overview: (if public or available)
- Latest revenue and profit figures
- Funding rounds, investors, and valuation (if startup)
- Stock performance (if publicly traded)

Recent News & Developments:
- Acquisitions, product launches, major announcements
- Legal or regulatory issues, if any

Competitors and Market Landscape:
- Key competitors
- Market share or positioning

Hiring and Culture:
- Open roles or hiring trends
- Company culture highlights (from Glassdoor, etc.)

Present the output in a clean, readable markdown format. If any information is not available, clearly indicate "Not Available" or "N/A".
"""

if submit_button and company_name:
    with st.spinner("Researching..."):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Tell me about {company_name} company?")
        ]

        final_output = []
        for step in agent_executor.stream({"messages": messages}, stream_mode="values"):
            final_output.append(step["messages"][-1].content)

        # Show the final markdown after streaming is complete
        st.markdown(final_output[-1], unsafe_allow_html=True)
        st.success("Research completed!")
        st.balloons()