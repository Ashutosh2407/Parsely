import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model = "llama-3.3-70b-versatile")
response = llm.invoke("what is a 10-K filling?")
print(response.content)