import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.state import ChatState
from src.tools.job_search import job_search
from src.tools.semantic_search import semantic_search
from src.tools.resume_match import resume_match

load_dotenv()

TOOLS = [job_search, semantic_search, resume_match]

SYSTEM_PROMPT = """
You are Toronto Job Market Assistant.

Your job is to help users:
- search Toronto-area job postings
- summarize market trends from available notes
- evaluate an uploaded resume against a target job role

Use tools whenever needed.
Be clear, practical, and concise.
If tool results are limited, say so honestly.
Do not invent facts beyond the available tool results.

When a user wants resume feedback, use the resume_match tool.
The resume_match tool expects:
- resume_file_path
- job_query
- location

"""

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0
).bind_tools(TOOLS)


def call_model(state: ChatState):
    messages = state["messages"]
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    return builder.compile()