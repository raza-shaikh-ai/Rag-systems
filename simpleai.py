from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

simple_prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the user's question clearly and concisely.
Answer in the SAME language as the user's question.
If you do not know the answer, say:
"I don't have enough information to answer that." (in the user's language)

Question:
{question}

Answer: 
""",
    input_variables=["question"]
)


model = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    api_key=os.getenv("openrouter").strip(),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500,
    temperature=0.4
)

parser = StrOutputParser()


simple_chain = simple_prompt | model | parser
