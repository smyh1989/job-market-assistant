from src.vectorstore.chroma_store import collection


results = collection.query(
    query_texts=["python sql dashboard"],
    n_results=3
)

print(results)