import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


@tool
def job_search(query: str, location: str = "Toronto", page: int = 1) -> str:
    """
    Search live job listings from Adzuna by keyword and location.
    """

    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return "Missing Adzuna API credentials. Add ADZUNA_APP_ID and ADZUNA_APP_KEY to your .env file."

    url = f"{BASE_URL}/ca/search/{page}"

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": 5,
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"API request failed: {e}"
    except ValueError:
        return "Could not parse the API response."

    results = data.get("results", [])

    if not results:
        return f"No jobs found for query: {query} in {location}"

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