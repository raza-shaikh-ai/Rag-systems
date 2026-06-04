from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

rag_prompt = PromptTemplate(
    template="""You are a precise assistant that answers questions using only the provided context.

INSTRUCTIONS:
- Give complete, readable answers
- Use ONLY information from the context below
- Answer in the SAME language as the user's question
- If the user asks for a name or date, keep that part brief
- If information is not in the context, respond: "Not in your documents."
- Do not repeat the question
- Use short paragraphs or bullet points when that makes the answer easier to read
- Use chat history to understand follow-up questions

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
    input_variables=["context", "question", "chat_history"]
)

model = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite",
    google_api_key=os.getenv("gemini"),
    max_output_tokens=800,
    temperature=0.1
)

parser = StrOutputParser()

rag_chain = rag_prompt | model | parser