import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def rewrite_job_query(user_query: str, default_location: str = "Toronto") -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
    "role": "system",
    "content": """
You extract job-search queries for a jobs API.

Return only valid json.
The output must be a json object.

Rules:
- Extract a short core job-title query as "title_query"
- Extract a broader keyword query as "full_query"
- Extract the city/region as "location_query"
- Do not include location words inside title_query or full_query
- Remove filler words
- Keep role title, seniority, skills, tools, certifications, and domain only if relevant
- title_query should be short and API-friendly, mainly the core role title
- full_query can include seniority, skills, tools, and domain
- If no location is mentioned, use the provided default location
- Do not invent details

- If the user explicitly mentions a location (city or region), you must extract that exact location.
- Never replace a mentioned location with the default location.
- The location must be extracted exactly from the user query when present.
- Do not invent details

json format:
{
  "title_query": "...",
  "full_query": "...",
  "location_query": "..."
}
"""     
           },
            {
    "role": "user",
    "content": f"""
Default location: {default_location}

User query:
{user_query}

Return only json.
"""
            }
        ]
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    title_query = data.get("title_query", "").strip()
    full_query = data.get("full_query", "").strip()
    location_query = data.get("location_query", default_location).strip() or default_location

    import re

    title_query = re.sub(r"[,\.;:]+", " ", title_query)
    title_query = re.sub(r"\s+", " ", title_query).strip()

    full_query = re.sub(r"[,\.;:]+", " ", full_query)
    full_query = re.sub(r"\s+", " ", full_query).strip()

    location_query = re.sub(r"[,\.;:]+", " ", location_query)
    location_query = re.sub(r"\s+", " ", location_query).strip()

    if not title_query:
        title_query = user_query
    if not full_query:
        full_query = title_query

    return {
        "title_query": title_query,
        "full_query": full_query,
        "location_query": location_query,
    }


def safe_rewrite_job_query(user_query: str, default_location: str = "Toronto") -> dict:
    try:
        rewritten = rewrite_job_query(user_query, default_location)

        return {
            "title_query": rewritten.get("title_query", user_query).strip() or user_query,
            "full_query": rewritten.get("full_query", user_query).strip() or user_query,
            "location_query": rewritten.get("location_query", default_location).strip() or default_location,
        }

    except Exception:
        return {
            "title_query": user_query,
            "full_query": user_query,
            "location_query": default_location,
        }


if __name__ == "__main__":
    tests = [
        "I want a junior data analyst job in Toronto with python skills",
        "entry level ux designer in Vancouver with figma",
        "remote project manager healthcare jobs",
        "show me nurse practitioner roles in calgary",
        "chef job toronto",
    ]

    for t in tests:
        print("\n" + "=" * 80)
        print("INPUT:", t)
        print(rewrite_job_query(t, default_location="Toronto"))

        