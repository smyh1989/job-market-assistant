import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from src.services.query_rewriter import safe_rewrite_job_query
from src.vectorstore.build_index import store_jobs_in_chroma

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


def fetch_jobs_from_adzuna(query: str, location: str = "Toronto", page: int = 1) -> list[dict]:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        raise ValueError("Missing Adzuna API credentials. Add ADZUNA_APP_ID and ADZUNA_APP_KEY to your .env file.")

    url = f"{BASE_URL}/ca/search/{page}"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": 5,
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    return data.get("results", [])


@tool
def job_search(query: str, location: str = "Toronto", page: int = 1) -> str:
    """
    Search live job listings from Adzuna by keyword and location.
    Automatically stores fetched results in ChromaDB for semantic search.
    """
    try:
        rewritten = safe_rewrite_job_query(query, default_location=location)

        title_query = rewritten["title_query"]
        full_query = rewritten["full_query"]
        clean_location = rewritten["location_query"]
        
        # remove wrong location
        invalid_locations = ["remote", "anywhere", "worldwide"]

        if clean_location.lower() in invalid_locations:
            clean_location = location  # fallback to default

        print("Original query:", query)
        print("Title query:", title_query)
        print("Full query:", full_query)
        print("Detected location:", clean_location)

        # a more advanced query
        results = fetch_jobs_from_adzuna(
            query=full_query,
            location=clean_location,
            page=page,
        )
        used_query = full_query

        # if the first query did not work!
        if not results and title_query != full_query:
            print("No results with full query. Trying title query...")
            results = fetch_jobs_from_adzuna(
                query=title_query,
                location=clean_location,
                page=page,
            )
            used_query = title_query

    except requests.RequestException as e:
        return f"API request failed: {e}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected error: {e}"

    if not results:
        return f"No jobs found for query: {query} in {clean_location}"

    try:
        msg = store_jobs_in_chroma(
            jobs=results,
            source_query=used_query,
            location=clean_location,
            page=page,
        )
        print(msg)
    except Exception as e:
        print("Chroma store error:", e)

    lines = []
    for job in results:
        title = job.get("title", "N/A")
        company = job.get("company", {}).get("display_name", "N/A")
        job_location = job.get("location", {}).get("display_name", "N/A")
        description = job.get("description", "No description available.")
        redirect_url = job.get("redirect_url", "No application link available.")

        lines.append(
            f"Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {job_location}\n"
            f"Description: {description[:300]}...\n"
            f"Apply: {redirect_url}"
        )

    return "\n\n---\n\n".join(lines)


if __name__ == "__main__":
    print("RUNNING MAIN...")

    test_queries = [
        "I want a junior data analyst job in Toronto with python skills",
        "entry level ux designer in Vancouver with figma",
        "remote project manager healthcare jobs",
        "show me nurse practitioner roles in calgary",
        "chef job toronto",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print("QUERY:", q)
        print("=" * 80)

        result = job_search.invoke({"query": q})
        print("\nRESULT:\n")
        print(result)