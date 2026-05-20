from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

simple_prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Question:
{question}

Answer: 
""",
    input_variables=["question"]
)

model = ChatOpenAI(
    model="deepseek/deepseek-v4-flash:free",
    api_key=os.getenv("openrouter").strip(),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500,
    temperature=0.4
)

parser = StrOutputParser()


simple_chain = simple_prompt | model | parser

res = simple_chain.invoke({"question": "hello how its going"})

print(res)
