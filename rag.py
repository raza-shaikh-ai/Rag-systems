from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv
import os

load_dotenv()

rag_prompt = PromptTemplate(
    template="""You are a precise assistant that answers questions using the provided context and chat history.

INSTRUCTIONS:
- Give complete, readable answers
- Use the CONTEXT and CHAT HISTORY together to answer the question
- For follow-up questions, refer to the chat history to understand what the user is asking about
- Answer in the SAME language as the user's question
- If the user asks for a name or date, keep that part brief
- Only say "Not in your documents." if the question is completely unrelated to both the context and chat history
- Do not repeat the question
- Use short paragraphs or bullet points when that makes the answer easier to read

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
    input_variables=["context", "question", "chat_history"]
)

model = ChatBedrockConverse(
    model="deepseek.v3.2",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    max_tokens=800,
    temperature=0.1
)

parser = StrOutputParser()

rag_chain = rag_prompt | model | parser