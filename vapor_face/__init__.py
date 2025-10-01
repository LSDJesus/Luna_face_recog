"""
VAPOR-FACE: Minimum Viable Proof-of-Concept
Semantic Facial Recognition with Surgical Vector Pruning

Core components for the Sovereign Persona Archive MVP.
"""

__version__ = "0.1.0-mvp"
__author__ = "VAPOR-FACE Team"

from .core.semantic_encoder import SemanticEncoder
from .core.vector_database import VectorDatabase
from .core.surgical_pruner import SurgicalPruner
from .gui.main_window import VaporFaceGUI

__all__ = [
    "SemanticEncoder",
    "VectorDatabase", 
    "SurgicalPruner",
    "VaporFaceGUI"
]