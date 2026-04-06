from typing import Any, List

from langchain_core.tools import tool

from src.services.resume_evaluator import (
    ResumeEvaluationError,
    evaluate_resume_with_llm,
    format_resume_match_result,
)
from src.services.resume_parser import ResumeParsingError, extract_resume_text
from src.services.query_rewriter import safe_rewrite_job_query
from src.tools.job_search import fetch_jobs_from_adzuna
from src.vectorstore.build_index import store_jobs_in_chroma
from src.vectorstore.chroma_store import get_vectorstore


MIN_RELEVANT_DOCS = 3


def retrieve_jobs_from_chroma(job_query: str, k: int = 5) -> List[Any]:
    """
    Retrieve relevant jobs from ChromaDB using LangChain similarity search.
    """
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(job_query, k=k)


def has_enough_context(docs: List[Any], min_docs: int = MIN_RELEVANT_DOCS) -> bool:
    """
    Check whether enough relevant documents were retrieved.
    """
    return len(docs) >= min_docs


def refresh_jobs_from_api(job_query: str, location: str = "Canada") -> list[dict]:
    """
    Fetch jobs from the API and store them in ChromaDB.
    """
    jobs = fetch_jobs_from_adzuna(query=job_query, location=location)

    if jobs:
        store_jobs_in_chroma(
            jobs=jobs,
            source_query=job_query,
            location=location,
            page=1,
        )

    return jobs


def evaluate_resume_against_jobs(
    resume_text: str,
    job_query: str,
    docs: List[Any],
    return_json: bool = False,
) -> str:
    """
    Evaluate the resume against retrieved job documents using the LLM evaluator.
    """
    result = evaluate_resume_with_llm(
        resume_text=resume_text,
        job_query=job_query,
        docs=docs,
    )

    if return_json:
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)

    return format_resume_match_result(result)


@tool
def resume_match(
    resume_file_path: str,
    job_query: str,
    location: str = "Canada",
    return_json: bool = False,
) -> str:
    """
    Evaluate a user's uploaded resume file against a target job.

    Args:
        resume_file_path: Path to the uploaded resume file (PDF or DOCX).
        job_query: Target role, for example 'data analyst' or 'UX designer'.
        location: Job market location to use if an API refresh is needed.
        return_json: If True, return structured JSON text.

    Returns:
        Resume match analysis as formatted text or JSON.
    """
    try:
        # 1) Parse the uploaded resume
        resume_text = extract_resume_text(resume_file_path)

        # 2) Rewrite / normalize the job query
        
        rewritten = safe_rewrite_job_query(job_query)
        rewritten_query = rewritten["full_query"]
        location = rewritten.get("location_query", location)

        # 3) Try Chroma first
        docs = retrieve_jobs_from_chroma(rewritten_query, k=5)

        # 4) If not enough context, fetch from API and store
        if not has_enough_context(docs):
            refresh_jobs_from_api(rewritten_query, location=location)
            docs = retrieve_jobs_from_chroma(rewritten_query, k=5)

        # 5) If still nothing useful, return a graceful message
        if not docs:
            return (
                "I could not find enough relevant job postings to evaluate this resume "
                f"for '{job_query}' in '{location}'."
            )

        # 6) Evaluate
        return evaluate_resume_against_jobs(
            resume_text=resume_text,
            job_query=rewritten_query,
            docs=docs,
            return_json=return_json,
        )

    except ResumeParsingError as e:
        return f"Resume parsing error: {e}"

    except ResumeEvaluationError as e:
        return f"Resume evaluation error: {e}"

    except Exception as e:
        return f"Unexpected error in resume_match: {e}"