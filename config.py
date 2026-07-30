import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_secret(key):

    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


MONGODB_URI = get_secret("MONGODB_URL")
GOOGLE_API_KEY = get_secret("GROQ_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")