
RAG Backend API

This project is a Retrieval-Augmented Generation (RAG) backend system that allows users to upload documents and ask questions based on their content.
It is designed for learning, interviews, and real-world use cases such as document assistants, internal knowledge bases, and AI search systems.

The backend is built using FastAPI, ChromaDB, Groq LLM, and HuggingFace embeddings, with full Docker support.

⸻

Key Features

Document Handling
	•	Supports multiple file formats:
	•	PDF
	•	DOCX
	•	CSV
	•	TXT
	•	Automatically processes uploaded documents
	•	Splits documents into chunks for efficient retrieval
	•	Adds metadata such as source and scope for isolation

Authentication and Security
	•	JWT-based authentication system
	•	Each user gets a unique data scope based on username
	•	Uploaded documents are isolated per user
	•	Tokens expire automatically after 4 days
	•	Temporary uploaded files are cleaned up after processing

Retrieval and Search
	•	Uses hybrid search:
	•	Vector similarity search (embeddings)
	•	BM25 keyword-based search
	•	Combines results to improve answer accuracy
	•	Efficient top-K document retrieval

AI Answer Generation
	•	Uses Groq LLM for fast and high-quality responses
	•	Context is generated from retrieved document chunks
	•	Designed for question answering over private documents
	•	Query rewriting is available but disabled by default for speed

Deployment Ready
	•	Fully Dockerized backend
	•	Easy local development setup
	•	Environment-based configuration
	•	Can be deployed to cloud platforms supporting Docker

⸻

Setup Instructions

Step 1: Clone the Repository and Install Dependencies

git clone <your-repo>
cd lang-project
pip install -r requirements.txt


⸻

Step 2: Environment Configuration

Create environment files from the provided examples:

cp .env.example .env
cp docker.env.example docker.env

Fill in the following values:
	•	JWT Key
Any secure random string (minimum 32 characters)
	•	Groq API Key
Obtain from the Groq Console
	•	HuggingFace Token
Generate from HuggingFace account settings

⸻

Step 3: Run Locally

python main.py

Open your browser and visit:

http://localhost:8000/docs

This provides interactive API documentation using Swagger UI.

⸻

Step 4: Run Using Docker

Build the Docker image:

docker build -t rag-backend .

Run the container using environment variables:

docker run -p 8000:8000 --env-file docker.env rag-backend


⸻

How to Use the API

Step 1: Login and Get Token

curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username"}'

This returns a JWT token.
This token is required for all protected endpoints.

⸻

Step 2: Upload a Document

curl -X POST "http://localhost:8000/inject/file" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@your_document.pdf"

The document is processed, chunked, embedded, and stored securely under your user scope.

⸻

Step 3: Ask Questions

curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

The system retrieves relevant chunks and generates an AI-based answer.

⸻

Supported File Types
	•	PDF files (.pdf)
	•	Word documents (.docx)
	•	CSV files (.csv)
	•	Text files (.txt)

⸻

API Endpoints Summary
	•	POST /login
Generates a JWT token
	•	POST /inject/file
Uploads and processes documents
	•	POST /ask
Answers questions based on uploaded documents
	•	GET /docs
Interactive API documentation

⸻

Performance Notes
	•	First startup may take time because embedding models are downloaded (approximately 2GB)
	•	After the first run, performance improves significantly
	•	Query rewriting is disabled by default to reduce latency
	•	Chunking and embeddings ensure efficient retrieval even for large documents

⸻

Docker Usage Examples

docker build -t rag-backend .

docker run -p 8000:8000 \
  -e key="your_jwt_key" \
  -e grok="your_groq_key" \
  rag-backend

docker run -p 8000:8000 --env-file docker.env rag-backend


⸻

Security Design
	•	User data is scoped and isolated
	•	JWT tokens are time-limited
	•	Documents are never shared across users
	•	File system cleanup prevents storage leaks

⸻

Common Issues and Fixes

Invalid token error
	•	Ensure JWT key matches the one used during login

Slow response on first run
	•	Model download happens only once

File upload failure
	•	Check file type and size

Docker container not starting
	•	Verify environment variables are correctly set

⸻

System Architecture Overview

User Request
   ↓
JWT Authentication
   ↓
Document Upload or Question
   ↓
Chunking and Embeddings
   ↓
Hybrid Retrieval (Vector + BM25)
   ↓
LLM Answer Generation
   ↓
Final Response


Tech Stack Used
	•	FastAPI for backend APIs
	•	ChromaDB for vector storage
	•	Groq for LLM inference
	•	HuggingFace for embeddings
	•	LangChain for RAG pipeline
	•	Docker for containerization
