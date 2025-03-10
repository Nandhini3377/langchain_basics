from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(
    temperature=0.6,
    model="gemini-1.5-flash",
    max_retries=2
)

# Initialize DuckDuckGo Search
def search_duckduckgo(query):
    search = DuckDuckGoSearchRun()
    return search.run(query)

# Define the Tool
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search_duckduckgo,
    description="Search the web for real-time information using DuckDuckGo."
)

# Create AI Agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Query the agent
response = agent.invoke("Current time in India?")
print("Assistant:", response)
