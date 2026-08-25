"""Embedding providers for dense retrieval.

The core package ships only the protocol and the callable adapter; concrete
providers live behind optional extras::

    pip install astro-anchor[openai]            # OpenAIEmbeddingProvider
    pip install astro-anchor[voyage]            # VoyageEmbeddingProvider
    pip install astro-anchor[local-embeddings]  # SentenceTransformerEmbeddingProvider
"""

from anchor.embeddings._base import CallableEmbeddingProvider
from anchor.embeddings.openai import OpenAIEmbeddingProvider
from anchor.embeddings.sentence_transformers import SentenceTransformerEmbeddingProvider
from anchor.embeddings.voyage import VoyageEmbeddingProvider

__all__ = [
    "CallableEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "VoyageEmbeddingProvider",
]
