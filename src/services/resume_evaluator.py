import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()


class ResumeEvaluationError(Exception):
    """Raised when resume evaluation fails."""


def build_job_context(docs: List[Any]) -> str:
    """
    Build a readable job context string from Chroma/LangChain documents.

    Args:
        docs: Retrieved job documents.

    Returns:
        Combined job context string.
    """
    parts = []

    for idx, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "") or ""

        title = metadata.get("title", "Unknown Title")
        company = metadata.get("company", "Unknown Company")
        location = metadata.get("location", "Unknown Location")
        salary = metadata.get("salary", "Not specified")

        part = (
            f"Job {idx}\n"
            f"Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Salary: {salary}\n"
            f"Description:\n{content}\n"
        )
        parts.append(part)

    return "\n" + ("\n" + ("-" * 60) + "\n").join(parts)


def build_resume_match_prompt(
    resume_text: str,
    job_query: str,
    job_context: str,
) -> str:
    """
    Build the prompt for resume-vs-job evaluation.
    """
    return f"""
You are an expert ATS-style resume evaluator and career assistant.

The user wants to apply for this target role:
{job_query}

Here are relevant real job postings retrieved from the system:
{job_context}

Here is the candidate's resume:
{resume_text}

Your task:
1. Evaluate how well the resume matches the target role.
2. Base your judgment primarily on the retrieved job requirements.
3. Identify skills and qualifications clearly demonstrated in the resume.
4. Identify missing skills, unclear evidence, or weak areas.
5. Suggest practical resume improvements tailored to this target role.
6. Be realistic and not overly generous.

Return ONLY valid JSON with this exact structure:
{{
  "match_score": 0,
  "matched_skills": [],
  "missing_skills": [],
  "strengths": [],
  "weaknesses": [],
  "summary": "",
  "suggestions": []
}}
""".strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON safely, including basic cleanup if the model wraps JSON in markdown.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ResumeEvaluationError(f"Model did not return valid JSON. Raw output: {text}") from e


def evaluate_resume_with_llm(
    resume_text: str,
    job_query: str,
    docs: List[Any],
) -> Dict[str, Any]:
    """
    Evaluate a resume against retrieved job postings.

    Args:
        resume_text: Extracted and cleaned resume text.
        job_query: Target role query.
        docs: Retrieved job documents.

    Returns:
        Parsed JSON evaluation result.

    Raises:
        ResumeEvaluationError: If model call fails or JSON is invalid.
    """
    if not docs:
        raise ResumeEvaluationError("No job documents provided for evaluation.")

    job_context = build_job_context(docs)
    prompt = build_resume_match_prompt(
        resume_text=resume_text,
        job_query=job_query,
        job_context=job_context,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise resume evaluator. "
                        "Always return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content
        if not content:
            raise ResumeEvaluationError("Empty response received from the model.")

        return _safe_json_loads(content)

    except Exception as e:
        if isinstance(e, ResumeEvaluationError):
            raise
        raise ResumeEvaluationError(f"Resume evaluation failed: {e}") from e


def format_resume_match_result(result: Dict[str, Any]) -> str:
    """
    Convert structured JSON result into readable text for end users.
    """
    match_score = result.get("match_score", "N/A")
    matched_skills = result.get("matched_skills", [])
    missing_skills = result.get("missing_skills", [])
    strengths = result.get("strengths", [])
    weaknesses = result.get("weaknesses", [])
    summary = result.get("summary", "")
    suggestions = result.get("suggestions", [])

    def format_list(items: List[str]) -> str:
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)

    return f"""
Match Score: {match_score}/100

Matched Skills:
{format_list(matched_skills)}

Missing Skills:
{format_list(missing_skills)}

Strengths:
{format_list(strengths)}

Weaknesses:
{format_list(weaknesses)}

Summary:
{summary}

Suggestions:
{format_list(suggestions)}
""".strip()