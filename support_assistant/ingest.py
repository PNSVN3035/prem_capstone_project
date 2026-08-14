from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"

CHROMA_DIR = BASE_DIR / "chroma_db"


#embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)

#chromadb client
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


COLLECTION_NAME = "zepto_policies"

try:
    client.delete_collection(
        COLLECTION_NAME
    )

    print(
        "Old collection removed."
    )

except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)

#load documents
document_files = sorted(
    DOCS_DIR.glob("doc_*.txt")
)


if len(document_files) != 8:

    raise ValueError(
        "Exactly 8 policy documents are required. "
        f"Found {len(document_files)}."
    )


documents = []
document_ids = []
metadatas = []


for file_path in document_files:

    text = file_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:

        raise ValueError(
            f"{file_path.name} is empty."
        )

    document_id = (
        file_path.stem
    )

    documents.append(
        text
    )

    document_ids.append(
        document_id
    )

    metadatas.append(
        {
            "source": file_path.name
        }
    )


print(
    f"Loaded {len(documents)} documents."
)

#chunking
chunks = documents

chunk_ids = [
    f"{doc_id}_chunk_01"
    for doc_id in document_ids
]


chunk_metadatas = [
    {
        "source": metadata["source"],
        "chunk_number": 1
    }
    for metadata in metadatas
]

#generate embeddings

print(
    "Generating embeddings..."
)


embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)


embeddings = embeddings.tolist()


print(
    f"Generated {len(embeddings)} embeddings."
)

#store in chromadb

collection.add(
    ids=chunk_ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=chunk_metadatas
)


print(
    "Documents stored in ChromaDB."
)

#verify collection
count = collection.count()


print(
    f"ChromaDB collection count: {count}"
)


if count != 8:

    raise ValueError(
        "Expected 8 stored chunks."
    )

#test retrieval
test_query = (
    "What is the delivery fee?"
)


query_embedding = (
    embedding_model.encode(
        [test_query],
        normalize_embeddings=True
    )
    .tolist()
)


results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)


print("\nTEST QUERY:")
print(test_query)


print("\nTOP 3 RETRIEVED SOURCES:")


for index in range(
    len(results["ids"][0])
):

    print(
        f"{index + 1}. "
        f"{results['metadatas'][0][index]['source']}"
    )

    print(
        results["documents"][0][index][:150]
    )

    print()