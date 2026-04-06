'''import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore.chroma_store import collection

import html
import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)

    replacements = {
        "u2019": "'",
        "u2014": " - ",
        "u2013": "-",
        "\xa0": " ",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_job_text(job: dict) -> str:
    title = clean_text(job.get("title", "N/A"))
    company = clean_text(job.get("company", {}).get("display_name", "N/A"))
    location = clean_text(job.get("location", {}).get("display_name", "N/A"))
    description = clean_text(job.get("description", "No description available."))
    category = clean_text(job.get("category", {}).get("label", "N/A"))
    redirect_url = job.get("redirect_url", "")

    return (
        f"Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Category: {category}\n"
        f"Apply: {redirect_url}\n"
        f"Description:\n{description}"
    )


def store_jobs_in_chroma(jobs: list[dict], source_query: str, location: str, page: int) -> str:
    if not jobs:
        return "No jobs to index."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    documents = []
    metadatas = []
    ids = []

    for job in jobs:
        full_text = build_job_text(job)
        chunks = splitter.split_text(full_text)

        title = job.get("title", "N/A")
        company = job.get("company", {}).get("display_name", "N/A")
        job_location = job.get("location", {}).get("display_name", "N/A")
        redirect_url = job.get("redirect_url", "")
        job_id = str(job.get("id", uuid.uuid4()))

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "job_id": job_id,
                "chunk_index": i,
                "title": title,
                "company": company,
                "location": job_location,
                "redirect_url": redirect_url,
                "source_query": source_query,
                "page": page,
            })
            ids.append(f"{job_id}_chunk_{i}")

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return f"Stored {len(jobs)} jobs and {len(documents)} chunks in ChromaDB." '''

import uuid
import html
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore.chroma_store import get_vectorstore


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)

    replacements = {
        "u2019": "'",
        "u2014": " - ",
        "u2013": "-",
        "\xa0": " ",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_job_text(job: dict) -> str:
    title = clean_text(job.get("title", "N/A"))
    company = clean_text(job.get("company", {}).get("display_name", "N/A"))
    location = clean_text(job.get("location", {}).get("display_name", "N/A"))
    description = clean_text(job.get("description", "No description available."))
    category = clean_text(job.get("category", {}).get("label", "N/A"))
    redirect_url = clean_text(job.get("redirect_url", ""))

    return (
        f"Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Category: {category}\n"
        f"Apply: {redirect_url}\n"
        f"Description:\n{description}"
    )


def build_documents_from_jobs(
    jobs: list[dict],
    source_query: str,
    location: str,
    page: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    documents: List[Document] = []

    for job in jobs:
        full_text = build_job_text(job)
        chunks = splitter.split_text(full_text)

        title = clean_text(job.get("title", "N/A"))
        company = clean_text(job.get("company", {}).get("display_name", "N/A"))
        job_location = clean_text(job.get("location", {}).get("display_name", "N/A"))
        redirect_url = clean_text(job.get("redirect_url", ""))
        category = clean_text(job.get("category", {}).get("label", "N/A"))
        job_id = str(job.get("id", uuid.uuid4()))

        for i, chunk in enumerate(chunks):
            metadata = {
                "job_id": job_id,
                "chunk_index": i,
                "title": title,
                "company": company,
                "location": job_location,
                "category": category,
                "redirect_url": redirect_url,
                "source_query": source_query,
                "search_location": location,
                "page": page,
            }

            documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                )
            )

    return documents


def store_jobs_in_chroma(
    jobs: list[dict],
    source_query: str,
    location: str,
    page: int,
) -> str:
    if not jobs:
        return "No jobs to index."

    vectorstore = get_vectorstore()
    documents = build_documents_from_jobs(
        jobs=jobs,
        source_query=source_query,
        location=location,
        page=page,
    )

    ids = [
        f"{doc.metadata['job_id']}_chunk_{doc.metadata['chunk_index']}"
        for doc in documents
    ]

    vectorstore.add_documents(
        documents=documents,
        ids=ids,
    )

    return f"Stored {len(jobs)} jobs and {len(documents)} chunks in ChromaDB."