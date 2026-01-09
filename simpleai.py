from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
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

model = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instant",
    api_key=os.getenv("openrouter"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=300
)

parser = StrOutputParser()

simple_chain = simple_prompt | model | parser

