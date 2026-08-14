from fastapi import FastAPI

from support_assistant.graph import support_graph
from support_assistant.schemas import AskRequest, AskResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "A policy support assistant using "
        "LangGraph, ChromaDB and local embeddings."
    ),
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running."
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
) -> AskResponse:

    result = support_graph.invoke(
        {
            "question": request.query
        }
    )

    response = AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )

    return response
