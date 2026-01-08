from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

rag_prompt = PromptTemplate(
    template="""
You are a precise question-answering assistant.

Answer the question using ONLY the context below.
The question may be incomplete — infer the intended question and answer it correctly.
If the answer is a person, return ONLY the person's name.
If the answer is not present in the context, say:
"Not in your documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("grok"),
    max_tokens= 150
)

parser = StrOutputParser()

rag_chain = rag_prompt | model | parser