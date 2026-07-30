from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


web_search_tool = TavilySearchResults(
    max_results=5
)