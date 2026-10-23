from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from config import GROQ_API_KEY
load_dotenv()

my_api_key=GROQ_API_KEY

def load_llm():
    """
    Load and return the ChatGroq LLM with automatic rate-limit retry handling.
    """
    llm = ChatGroq(
        model='qwen/qwen3.8-27b',
        api_key=my_api_key,
        temperature=0,
        max_retries=5
    )

    return llm


