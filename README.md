🚀 Toronto Job Market Assistant (AI + RAG + Resume Evaluator)

An AI-powered assistant that helps users explore the Toronto job market, perform semantic search over job postings, and evaluate resumes against real-world job requirements.

This project combines Retrieval-Augmented Generation (RAG), vector databases (Chroma), and LLM-based reasoning to provide practical, data-driven insights for job seekers.

---

FEATURES

Job Search (API + Storage)
- Search real job postings using the Adzuna API
- Automatically store results in ChromaDB
- Build a growing local job knowledge base

Semantic Search (RAG)
- Ask high-level questions like:
  "What skills are most common for data analyst jobs?"
- Retrieve relevant job chunks using vector similarity
- Generate insights using an LLM

Resume Match (AI Evaluation)
- Upload a resume (PDF or DOCX)
- Compare it against real job postings
- Get:
  Match score
  Matched skills
  Missing skills
  Strengths & weaknesses
  Actionable suggestions

Chat-based Interaction
- Natural language interface powered by LangGraph
- Automatic tool selection (job search, semantic search, resume match)

---

ARCHITECTURE

User → Gradio UI  
      ↓  
LangGraph Agent (LLM + Tools)  
      ↓  
Tools:
- job_search → API + store in Chroma
- semantic_search → retrieve from Chroma
- resume_match → parse + retrieve + evaluate  

Vector Store:
- ChromaDB (LangChain integration)

LLM:
- OpenAI (GPT-4o-mini)

---

TECH STACK

- Python 3.11+
- LangChain
- LangGraph
- ChromaDB
- OpenAI API
- Gradio (UI)
- Adzuna Job API

---

INSTALLATION

git clone https://github.com/YOUR_USERNAME/toronto-job-assistant.git
cd toronto-job-assistant

python -m venv .venv

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

pip install -e .

---

ENVIRONMENT VARIABLES

OPENAI_API_KEY=your_key_here
ADZUNA_APP_ID=your_id
ADZUNA_APP_KEY=your_key

---

RUN THE APP

python app.py

Open:
http://127.0.0.1:7860

---

EXAMPLE USE CASES

Job Search:
Find data analyst jobs in Toronto

Semantic Search:
What skills are most common in those jobs?

Resume Match:
Upload resume → enter "data analyst" → get evaluation

---

SAMPLE OUTPUT

Match Score: 65/100

Matched Skills:
- SQL
- Python
- Excel
- Tableau

Missing Skills:
- Large dataset experience
- Power BI

---

LIMITATIONS

- Semantic answers may be partially generalized
- PDF parsing depends on formatting
- Results depend on API availability

---

FUTURE IMPROVEMENTS

- Skill frequency extraction
- Better semantic grounding
- Improved UI/UX
- Personalized resume suggestions

---

AUTHOR

Somi Shafiee  
Toronto, Canada  
LinkedIn: https://www.linkedin.com/in/somi-shafiee89/

