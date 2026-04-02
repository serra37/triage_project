import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "medical_triage"

def get_chroma_client():
    """Returns a Chroma vector store instance."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    return db

def get_retriever(k=3):
    """Returns a LangChain retriever interface for the vector store."""
    db = get_chroma_client()
    return db.as_retriever(search_kwargs={"k": k})
