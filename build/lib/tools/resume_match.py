from langchain_core.tools import tool


@tool
def resume_match(resume_text: str, job_description: str) -> str:
    """
    Compare resume text with a job description using simple keyword overlap.
    """
    resume_words = set(word.lower().strip(".,:;()[]") for word in resume_text.split())
    job_words = set(word.lower().strip(".,:;()[]") for word in job_description.split())

    important_words = {
        w for w in job_words
        if len(w) > 3 and w not in {
            "with", "that", "this", "from", "have", "will", "your",
            "about", "role", "team", "work", "data"
        }
    }

    overlap = sorted(list(important_words.intersection(resume_words)))
    missing = sorted(list(important_words.difference(resume_words)))[:15]

    score = 0
    if important_words:
        score = int((len(overlap) / len(important_words)) * 100)

    return (
        f"Estimated match score: {score}%\n\n"
        f"Matched keywords: {', '.join(overlap) if overlap else 'None'}\n\n"
        f"Potential missing keywords: {', '.join(missing) if missing else 'None'}"
    )