from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from config import GROQ_API_KEY
load_dotenv()

my_api_key=GROQ_API_KEY

def load_llm():
    """
    Load and return the ChatGroq LLM.
    """
    llm=ChatGroq(
        model='llama-3.3-70b-versatile',
        api_key=my_api_key,
        temperature=0
    )

    return llm

