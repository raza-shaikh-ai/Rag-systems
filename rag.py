from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

rag_prompt = PromptTemplate(
    template="""You are a precise assistant that answers questions using only the provided context.

INSTRUCTIONS:
- Give direct, concise answers
- Use ONLY information from the context below
- If asking for a name, respond with just the name
- If asking for a date, respond with just the date
- If information is not in the context, respond: "Not in your documents."
- Do not repeat the question
- Do not add extra explanations

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
    input_variables=["context", "question"]
)

model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("openrouter").strip(),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=150,
    temperature=0.1
)

parser = StrOutputParser()

rag_chain = rag_prompt | model | parser