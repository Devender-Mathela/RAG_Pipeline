from dotenv import load_dotenv
from config import TAVILY_API_KEY
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

tavily_api_key=TAVILY_API_KEY

web_search_tool = TavilySearchResults(
    tavily_api_key=tavily_api_key,
    max_results=5
)