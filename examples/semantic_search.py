#!/usr/bin/env python3
"""Semantic search example - user provides their own embedding model"""

from shodh_memory import Memory

# User brings their own embedding model (any model they want)
from sentence_transformers import SentenceTransformer

def main():
    # Initialize user's embedding model
    print("📦 Loading embedding model (user's choice)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # User's choice

    # Initialize memory
    memory = Memory(user_id="semantic_demo")

    # Add memories WITH embeddings
    print("\n📝 Adding memories with embeddings...")

    documents = [
        "I love Python programming and machine learning",
        "Rust is great for systems programming",
        "JavaScript is essential for web development",
        "Deep learning models require lots of data",
        "Building REST APIs with FastAPI is easy"
    ]

    for doc in documents:
        # User generates embedding with their model
        embedding = model.encode(doc)

        # Shodh-Memory just stores it
        memory.add(
            content=doc,
            experience_type="learning",
            embeddings=embedding.tolist()  # Convert numpy to list
        )

    print(f"✅ Added {len(documents)} memories with embeddings")

    # Semantic search - find by meaning, not keywords
    print("\n🔍 Semantic Search Examples:")

    queries = [
        "favorite programming languages",  # No keyword match but semantically similar
        "artificial intelligence training",  # AI = deep learning
        "web frameworks"  # web = JavaScript/FastAPI
    ]

    for query_text in queries:
        print(f"\n📍 Query: '{query_text}'")

        # User generates query embedding
        query_embedding = model.encode(query_text)

        # Semantic search
        results = memory.search(
            query_embedding=query_embedding.tolist(),
            max_results=2
        )

        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. [{result['importance']:.2f}] {result['experience']['content']}")

    # Hybrid search - both keyword AND semantic
    print("\n🔄 Hybrid Search (keyword + semantic):")
    query_text = "Python"
    query_embedding = model.encode(query_text)

    results = memory.search(
        query=query_text,  # Keyword filter
        query_embedding=query_embedding.tolist(),  # + Semantic ranking
        max_results=3
    )

    print(f"Results for '{query_text}' (hybrid):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['experience']['content']}")

if __name__ == "__main__":
    main()
