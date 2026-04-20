"""
data/vectorstore/

Vector store abstraction layer.

Backends:
    faiss    → FAISSVectorStore    (dev, default)
    pinecone → PineconeVectorStore (production)

Swap by changing VECTOR_STORE_TYPE in config.py.
"""