from langchain_core.tools import tool
from src.vectorstore.chroma_store import get_vectorstore


@tool
def semantic_search(query: str, n_results: int = 5) -> str:
    """
    Search semantically across stored job-description chunks in ChromaDB.
    """
    try:
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=n_results)
    except Exception as e:
        return f"Semantic search failed: {e}"

    if not docs:
        return "No indexed job data found yet. Run a job search first."

    lines = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        title = meta.get("title", "N/A")
        company = meta.get("company", "N/A")
        location = meta.get("location", "N/A")
        link = meta.get("redirect_url", "")

        lines.append(
            f"Result {i}\n"
            f"Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Relevant text: {doc.page_content}\n"
            f"Apply: {link}"
        )

    return "\n\n---\n\n".join(lines)