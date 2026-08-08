"""GreenRAG retrieval package: pluggable local / Azure vector stores."""

from .base import VectorStore
from .factory import get_vector_store

__all__ = ["VectorStore", "get_vector_store"]
