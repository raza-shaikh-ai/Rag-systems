from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

simple_prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the user's question clearly and concisely.
If you do not know the answer, say:
"I don't have enough information to answer that."

Question:
{question}

Answer:
""",
    input_variables=["question"]
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("grok"),
    max_tokens=150
)

parser = StrOutputParser()

simple_chain = simple_prompt | model | parser

