"""RAG utilities for encoding and scoring documents."""

__all__ = ["encode", "cos_sim"]

from .encoders import encode, cos_sim
