from pathlib import Path
from langchain_core.tools import tool

NOTES_PATH = Path("data/toronto_market_notes.txt")


@tool
def semantic_search(query: str) -> str:
    """
    Search curated Toronto job-market notes.
    This is a simple placeholder for future vector search.
    """
    if not NOTES_PATH.exists():
        return "Market notes file not found."

    text = NOTES_PATH.read_text(encoding="utf-8")
    query_words = [w.lower() for w in query.split() if w.strip()]

    sentences = [s.strip() for s in text.split(".") if s.strip()]
    matches = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for w in query_words if w in sentence_lower)
        if score > 0:
            matches.append((score, sentence))

    if not matches:
        return "No relevant market notes found."

    matches.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [item[1] for item in matches[:3]]
    return ". ".join(top_sentences) + "."