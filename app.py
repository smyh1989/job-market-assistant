import os
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import build_graph
from src.tools.resume_match import resume_match
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)
graph = build_graph()


def course_chat(message, history):
    langchain_messages = []

    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    langchain_messages.append(HumanMessage(content=message))

    state = {"messages": langchain_messages}
    response = graph.invoke(state)

    return response["messages"][-1].content


def evaluate_resume(resume_file, job_query, location):
    if resume_file is None:
        return "Please upload a resume file."

    if not job_query.strip():
        return "Please enter a target job title or query."

    file_path = resume_file.name

    result = resume_match.invoke({
        "resume_file_path": file_path,
        "job_query": job_query,
        "location": location or "Toronto",
        "return_json": False,
    })

    return result


with gr.Blocks(title="Toronto Job Market Assistant") as demo:
    gr.Markdown("# Toronto Job Market Assistant")
    gr.Markdown(
        "Search Toronto jobs, ask semantic questions about stored job data, "
        "or upload a resume and evaluate it against a target role."
    )

    with gr.Tab("Chat"):
        chatbot = gr.ChatInterface(
            fn=course_chat,
            title="Chat Assistant",
            description="Ask about Toronto data jobs, market notes, or job trends."
        )

    with gr.Tab("Resume Match"):
        resume_file = gr.File(
            label="Upload Resume (PDF or DOCX)",
            file_types=[".pdf", ".docx"]
        )
        job_query = gr.Textbox(
            label="Target Job",
            placeholder="e.g. data analyst"
        )
        location = gr.Textbox(
            label="Location",
            value="Toronto"
        )
        output = gr.Textbox(
            label="Resume Match Result",
            lines=18
        )
        run_button = gr.Button("Evaluate Resume")

        run_button.click(
            fn=evaluate_resume,
            inputs=[resume_file, job_query, location],
            outputs=output,
        )


if __name__ == "__main__":
    logger.info("Starting Toronto Job Market Assistant...")
    demo.launch()