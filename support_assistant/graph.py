import os
from pathlib import Path
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END


BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"

MODEL_NAME = "all-MiniLM-L6-v2"

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

#load embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)

#connect to chromadb
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

#graph state
class SupportState(TypedDict, total=False):

    question: str

    intent: str

    answer: str

    sources: list[str]

    confidence: float

#policy keywords
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

#classify intent

def classify_intent(state: SupportState):

    question = state["question"].lower()

    is_policy_question = any(
        keyword in question
        for keyword in POLICY_KEYWORDS
    )

    if is_policy_question:
        return {
            "intent": "policy_question"
        }

    return {
        "intent": "general_question"
    }


#mock answer
def build_mock_policy_answer(
    question: str,
    documents: list[str],
):

    question_lower = question.lower()

    context = " ".join(documents)

    if (
        "delivery fee" in question_lower
        or "delivery cost" in question_lower
        or "free delivery" in question_lower
    ):

        return (
            "Standard delivery is free on orders over "
            "INR 149. Orders below INR 149 have a flat "
            "INR 25 delivery fee. Priority delivery costs "
            "an additional INR 15."
        )

    if (
        "cancel" in question_lower
        or "cancellation" in question_lower
    ):

        return (
            "An order can be cancelled free of cost before "
            "its status changes to Packed. Once the order "
            "is packed, it cannot be cancelled through the app."
        )

    if (
        "return" in question_lower
        or "refund" in question_lower
    ):

        return (
            "Grocery and perishable items can be reported "
            "for return within 24 hours if they are damaged, "
            "spoiled, or incorrect. Approved refunds are "
            "usually credited to the original payment method "
            "within 3 to 5 business days."
        )

    if (
        "gift card" in question_lower
    ):

        return (
            "Zepto gift cards are available in INR 100, "
            "INR 250, INR 500, and INR 1000 denominations. "
            "They are valid for one year from the date of issue."
        )

    if (
        "support" in question_lower
        or "phone" in question_lower
        or "chat" in question_lower
    ):

        return (
            "Zepto in-app chat support is available 24/7. "
            "Email support is available for non-urgent queries. "
            "Phone support is not offered."
        )

    return (
        "I found relevant Zepto policy information for your "
        "question. Please check the provided source document "
        "for the exact policy details."
    )

#retrieve & answer

def retrieve_and_answer(state: SupportState):

    question = state["question"]

    query_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True,
        )
        .tolist()
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    sources = []

    for item in metadata:

        source = item.get(
            "source",
            "unknown"
        )

        if source not in sources:
            sources.append(source)

    top_chunk = documents[0]

    top_chunk_snippet = top_chunk[:200]

    answer = (
        f"Based on the retrieved context: "
        f"{top_chunk_snippet}"
    )

    confidence = 1.0


    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


#direct answer
def direct_answer(state: SupportState):

    answer = (
        "I can only answer questions about "
        "Zepto policies right now."
    )

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
    }

#conditional routing
def route_question(
    state: SupportState,
):

    if state["intent"] == "policy_question":

        return "policy_question"

    return "general_question"

#create langgraph
workflow = StateGraph(
    SupportState
)


workflow.add_node(
    "classify_intent",
    classify_intent,
)

workflow.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

workflow.add_node(
    "direct_answer",
    direct_answer,
)


workflow.add_edge(
    START,
    "classify_intent",
)


workflow.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)


workflow.add_edge(
    "retrieve_and_answer",
    END,
)

workflow.add_edge(
    "direct_answer",
    END,
)


support_graph = workflow.compile()

#test graph
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("POLICY QUESTION TEST")
    print("=" * 60)

    policy_result = support_graph.invoke(
        {
            "question":
                "What is the delivery fee?"
        }
    )

    print(
        "Intent:",
        policy_result["intent"]
    )

    print(
        "Answer:",
        policy_result["answer"]
    )

    print(
        "Sources:",
        policy_result["sources"]
    )

    print(
        "Confidence:",
        policy_result["confidence"]
    )


    print("\n" + "=" * 60)
    print("GENERAL QUESTION TEST")
    print("=" * 60)

    general_result = support_graph.invoke(
        {
            "question":
                "What is machine learning?"
        }
    )

    print(
        "Intent:",
        general_result["intent"]
    )

    print(
        "Answer:",
        general_result["answer"]
    )

    print(
        "Sources:",
        general_result["sources"]
    )

    print(
        "Confidence:",
        general_result["confidence"]
    )

