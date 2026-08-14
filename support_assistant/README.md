# Module 3 — Zepto Support Assistant

## Overview

In this module, I built a simple Zepto support assistant that answers questions using the policy documents provided in the assignment.

The main idea is to take the policy documents, convert them into embeddings, store them in ChromaDB, and then retrieve the most relevant policy when a user asks a question.

I used LangGraph to control the flow of the question and FastAPI to expose the assistant through a `/ask` API endpoint.

The graded version works in mock mode, so it does not require any paid LLM service or API key.

---

## Main Files

The main files in this module are:

* `docs/` — contains the 8 Zepto policy documents
* `ingest.py` — loads the policy files, creates embeddings, and stores them in ChromaDB
* `graph.py` — contains the LangGraph flow and retrieval logic
* `schemas.py` — contains the Pydantic request and response models
* `main.py` — contains the FastAPI application
* `Dockerfile` — runs the FastAPI service inside Docker
* `chroma_db/` — stores the local ChromaDB collection

---

## Policy Documents

I created 8 text files using the policy content provided in the assignment.

The documents cover:

* Delivery policy
* Returns and refunds
* Membership tiers
* Order tracking
* Order cancellation
* Damaged or missing items
* Gift cards
* Customer support hours

The documents are short, so I used one document as one chunk.

This gives:

```text
8 documents
→ 8 chunks
→ 8 embeddings
```

---

## Embeddings

I used the `all-MiniLM-L6-v2` model from `sentence-transformers`.

The model converts each policy document into a numeric embedding.

These embeddings are stored in a ChromaDB collection called:

```text
zepto_policies
```

This step runs locally and does not need an LLM API key.

---

## ChromaDB Retrieval

When a policy question is asked, the question is also converted into an embedding.

ChromaDB compares the query embedding with the stored policy embeddings and returns the top 3 most similar chunks.

For example, I tested:

```text
What is the delivery fee?
```

The top result was:

```text
doc_01.txt
```

This is correct because `doc_01.txt` contains the delivery policy.

---

## LangGraph Flow

I created a LangGraph `StateGraph` with three main nodes.

### classify_intent

This node decides whether the question is related to a Zepto policy.

The mock-mode keywords include:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

If one of these keywords is found, the question is classified as:

```text
policy_question
```

Otherwise it is classified as:

```text
general_question
```

### retrieve_and_answer

Policy questions are sent to this node.

This node:

1. creates an embedding for the question
2. retrieves the top 3 matching policy chunks from ChromaDB
3. uses the most relevant chunk to create the answer

In mock mode, the answer starts with:

```text
Based on the retrieved context:
```

### direct_answer

General questions are sent to this node.

In mock mode, the answer is:

```text
I can only answer questions about Zepto policies right now.
```

No ChromaDB retrieval is needed for this route.

---

## Mock Mode

The graded version uses mock mode.

If `MOCK_LLM` is not set, or if it is set to:

```text
MOCK_LLM=1
```

the application runs without making any real LLM API call.

In mock mode:

* intent classification uses the keyword rule
* policy questions use ChromaDB retrieval
* the answer is created from the top retrieved chunk
* general questions use a fixed response
* confidence is set to `1.0`
* no API key is required

The real LLM path is optional and is not needed for the graded submission.

---

## Structured Prompt

For the optional real LLM path, I created a prompt with the required structure.

```text
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the retrieved Zepto policy information.

TASK:
Answer the customer's question accurately.

FORMAT:
Return a clear answer based on the given policy context.

LENGTH:
Keep the answer short and easy to understand.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.

FEW-SHOT EXAMPLE:

Question:
What is the delivery fee?

Context:
Standard delivery is free on orders over INR 149.
Orders below INR 149 have a flat INR 25 delivery fee.

Answer:
Standard delivery is free on orders over INR 149.
For orders below INR 149, the delivery fee is INR 25.
```

---

## JSON Response Format

The API response follows this structure:

```json
{
  "answer": "string",
  "sources": [],
  "confidence": 1.0
}
```

The fields are:

* `answer` — the response shown to the user
* `sources` — the policy files used for the answer
* `confidence` — a value between 0 and 1

Pydantic is used to validate this structure.

---

## FastAPI

The application has a FastAPI endpoint:

```text
POST /ask
```

The request format is:

```json
{
  "query": "What is the delivery fee?"
}
```

---

## Example 1 — Policy Question

Request:

```json
{
  "query": "What is the delivery fee?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01.txt",
    "doc_05.txt",
    "doc_02.txt"
  ],
  "confidence": 1.0
}
```

This question is classified as `policy_question`, so ChromaDB retrieval is used.

---

## Example 2 — General Question

Request:

```json
{
  "query": "What is machine learning?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This question is classified as `general_question`, so no policy retrieval is needed.

---

## RAG Architecture

The support assistant follows four main steps:

```text
Ingestion
   ↓
Embedding
   ↓
Retrieval
   ↓
Generation
```

### Ingestion

`ingest.py` reads the 8 policy files from the `docs` folder.

Since the documents are short, each document is treated as one chunk.

### Embedding

`ingest.py` uses `all-MiniLM-L6-v2` to create embeddings for the policy chunks.

The embeddings are stored in the ChromaDB collection:

```text
zepto_policies
```

### Retrieval

When a policy question reaches `retrieve_and_answer`, the question is converted into an embedding.

ChromaDB compares it with the stored policy embeddings and returns the top 3 most similar chunks.

### Generation

In mock mode, no real LLM is used.

For a policy question, the answer is created from the most relevant retrieved chunk.

For a general question, the assistant returns a fixed message.

The optional real-LLM mode would use the structured prompt and retrieved context to generate the answer.

---

## How to Run

Install the project dependencies from the root folder:

```bash
pip install -r requirements.txt
```

Run the ingestion step:

```bash
python3 support_assistant/ingest.py
```

Start the FastAPI application:

```bash
MOCK_LLM=1 uvicorn support_assistant.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

The `/ask` endpoint can be tested from the FastAPI page.

---

## Docker

I created a Dockerfile for the FastAPI application.

Build the Docker image from the project root:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
```

Run the container:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

Then open:

```text
http://127.0.0.1:7860/docs
```

The Zepto support assistant can be tested from the same `/ask` endpoint.

---

## Final Summary

In this module, I built a small RAG-based support assistant for Zepto policies.

The policy documents are stored locally, converted into embeddings, and saved in ChromaDB. LangGraph decides whether a question needs policy retrieval or a direct response. FastAPI exposes the workflow through the `/ask` endpoint.

The required version works completely in mock mode, so no paid LLM service or API key is needed.
