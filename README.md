# RAG Document Assistant - Backend

A production-ready Retrieval-Augmented Generation (RAG) application backend that enables users to query their documents using their own LLM API keys, built with an agentic AI workflow and cloud storage integration. The project is deployed [here](https://rag.joshkjx.com). Signups currently disabled, but demo credentials available on request.

# Motivation

While using tools like Google's [NotebookLM](https://notebooklm.google/) for studying during my Master's program, I frequently hit usage limits and found myself frustrated by the inability to just switch to a different provider and continue my revision.
Driven by that experience, this project seeks to create a flexible broswer-based RAG system where users can bring their own API keys, giving them control over costs and provider choice while maintaining the convenience of a hosted application.

# Key Features

## Core Functionality

**Bring Your Own Key (BYOK)**: Support for multiple LLM providers (Currently OpenAI and Anthropic) - users control their own API usage and costs

**Agentic RAG Workflow**: LangChain-based pipeline with intelligent query rewriting and relevance checking

 - Conditional query rewriting node that reformulates queries for better retrieval
 - Post-retrieval relevance check that triggers re-querying if results are insufficient

**Streaming Responses**: Asynchronous streaming of LLM responses for improved UX

**Persistent Document Storage**: Cloud-based document management

**Full CRUD Operations**: Upload, retrieve, update, and delete documents with proper user isolation

## Security & Authentication

**JWT-based Authentication**: Secure access/refresh token pattern using httpOnly cookies

**Password Encryption**: Argon2 password hashing for secure credential storage

## Flexibility & Scalability

**Storage Abstraction**: S3-compatible interface allows easy migration between storage providers

**Stateless Architecture**: Session state managed in conversation history for horizontal scaling



# Tech Stack
## Core Framework:

FastAPI (REST API and async request handling)

SQLAlchemy (ORM and database management)

LangChain (agentic AI orchestration)

## AI/ML:

HuggingFace Transformers (embedding generation)

ChromaDB (vector storage)

# Setup
## Prerequisites

Python 3.11

Docker

PostgreSQL

Cloud Storage access

OpenAI/Anthropic API keys

## Installation
```bash
# Clone the repository
git clone https://github.com/joshkjx/rag-byok-backend
cd rag-byok-backend

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration before moving to the next step

# build the image
docker compose build

```

## Running the Application

### Start the server
```bash
# run the image
docker compose up -d
# API will be available at http://localhost:8080/api

# To access logs:
docker compose logs -f
```

## API Endpoints

### Authentication
- `POST /auth/signup` - Create new user account
- `POST /auth/login` - Login and receive JWT tokens
- `POST /auth/refresh` - Refresh access token

### Documents
- `GET /documents/get` - List user's documents
- `POST /documents/upload` - Upload document for embedding and storage. If a record of the document already exists, the existing document is overwritten.
- `DELETE /documents/delete` - Delete document - document ID expected in request body

### RAG Query
- `POST /query` - Submit query and receive streamed response from RAG agent

## Roadmap

- [ ] Rate limiting for API endpoints
- [ ] Cost dashboard with token usage tracking
- [ ] Support for additional LLM providers

# Project Status

This project is actively maintained and deployed as a demo application. It was built as a learning project to explore agentic AI deployment, cloud infrastructure, and production-grade API development. 
