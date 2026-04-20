"""
data/vectorstore/faiss_store.py

FAISS vector store implementation.

Wraps the FAISS index used in retrieval.py behind the
BaseVectorStore interface for easy swapping.

Currently retrieval.py uses FAISS directly — this wrapper
will be wired in when the abstraction layer is activated.
"""

# Implementation planned — retrieval.py uses FAISS directly for now.