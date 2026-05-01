import os
import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import cohere

co = cohere.Client(os.getenv("COHERE_API_KEY"))

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "medical_triage"

# Global instantiations to avoid loading models and index repeatedly during execution.
_db_instance = None
_bm25_instance = None
_bm25_docs = None

def get_chroma_client():
    """Returns a Chroma vector store instance."""
    global _db_instance
    if _db_instance is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        _db_instance = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )
    return _db_instance

def get_bm25_client():
    """Returns an in-memory BM25 index built from all ChromaDB documents."""
    global _bm25_instance, _bm25_docs
    if _bm25_instance is None:
        db = get_chroma_client()
        # Fetch all documents to build BM25 in-memory
        all_data = db.get()
        _bm25_docs = []
        tokenized_corpus = []
        for i in range(len(all_data['ids'])):
            text = all_data['documents'][i]
            metadata = all_data['metadatas'][i]
            _bm25_docs.append(Document(page_content=text, metadata=metadata))
            tokenized_corpus.append(text.lower().split())
        
        if tokenized_corpus:
            _bm25_instance = BM25Okapi(tokenized_corpus)
        else:
            _bm25_instance = BM25Okapi([["_empty_"]]) # Fallback for empty DB
    return _bm25_instance, _bm25_docs


def rrf_score(rank, k=60):
    return 1 / (k + rank)

def hybrid_search_with_rrf(query: str, k: int = 20):
    """Combines Dense (Chroma) and Sparse (BM25) search results using RRF."""
    db = get_chroma_client()
    bm25, bm25_docs = get_bm25_client()
    
    if not bm25_docs:
        return []

    # 1. Dense Search (BGE-M3)
    dense_results = db.similarity_search(query, k=k)
    
    # 2. Sparse Search (BM25)
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)
    
    # Get top indices
    top_sparse_idx = np.argsort(sparse_scores)[::-1][:k]
    sparse_results = [bm25_docs[i] for i in top_sparse_idx]
    
    # 3. RRF Fusion
    def get_doc_key(doc):
        # Using string representation of metadata + content as unique key
        return f"{doc.page_content}_{str(doc.metadata)}"
    
    doc_scores = {}
    doc_map = {}
    
    # Score Dense
    for rank, doc in enumerate(dense_results, start=1):
        key = get_doc_key(doc)
        doc_map[key] = doc
        doc_scores[key] = doc_scores.get(key, 0) + rrf_score(rank)
        
    # Score Sparse
    for rank, doc in enumerate(sparse_results, start=1):
        key = get_doc_key(doc)
        doc_map[key] = doc
        doc_scores[key] = doc_scores.get(key, 0) + rrf_score(rank)
        
    # Sort docs by combined RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_10_docs = [doc_map[key] for key, score in sorted_docs[:k]]
    
    return top_10_docs

def rerank_documents(query: str, docs: list, top_k: int = 5):
    """Reranks a list of documents against the query using Cohere."""
    if not docs:
        return []
    results = co.rerank(
        query=query,
        documents=[d.page_content for d in docs],
        top_n=top_k,
        model="rerank-english-v3.0"
    )
    return [docs[r.index] for r in results.results]
def search_and_rerank(query: str, k: int = 20, final_k: int = 5):
    """Single wrapper to perform Hybrid RRF Search followed by Reranking."""
    top_hybrid_docs = hybrid_search_with_rrf(query, k=k)
    final_docs = rerank_documents(query, top_hybrid_docs, top_k=final_k)
    return final_docs
