'''import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"), "...")

collection = client.get_or_create_collection(
    name="toronto_jobs",
    embedding_function=embedding_function
)
'''

import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "toronto_jobs"


def get_embedding_function() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )