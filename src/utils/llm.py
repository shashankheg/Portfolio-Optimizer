import os 
from dotenv import  load_dotenv
from langchain_groq import ChatGroq


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__),"../../.env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

def get_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
    '''Model for all the complicated tasks that require a high level of understanding and reasoning.'''
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=GROQ_API_KEY
    )

def get_fast_llm():
    """Faster model for simple tasks like sentiment scoring."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )