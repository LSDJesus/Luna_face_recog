"""
ArcFace Encoder for VAPOR-FACE MVP
===================================
Real face recognition embeddings using InsightFace's buffalo_l model pack.

This encoder replaces the mock encoder with actual ArcFace embeddings,
enabling real validation of semantic archaeology hypotheses on production FR vectors.

Author: Brian & Luna
Created: 2025-10-22
"""

import numpy as np
from typing import Optional, Dict, Any
import torch
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not available. Install with: pip install insightface")


class ArcFaceEncoder:
    """
    ArcFace encoder using InsightFace for 512D identity embeddings.
    
    This is the real deal—production-quality FR embeddings that we'll
    surgically probe for semantic axes. Uses buffalo_l model pack by default
    (98.5% LFW accuracy), with fallback to antelopev2 if needed.
    
    Implements SemanticEncoder interface for compatibility with VAPOR-FACE MVP.
    
    Attributes:
        model_name: InsightFace model pack name
        device: Compute device (cuda/cpu)
        app: FaceAnalysis instance
        embedding_dim: Always 512D (ArcFace standard)
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        det_size: tuple = (640, 640),
        **kwargs
    ):
        """
        Initialize ArcFace encoder with InsightFace.
        
        Args:
            model_name: Model pack name (buffalo_l, antelopev2, buffalo_sc)
            device: Device for inference
            det_size: Detection input size (larger = slower but more accurate)
            **kwargs: Additional FaceAnalysis arguments
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "InsightFace required for ArcFace encoder. "
                "Install with: pip install insightface"
            )
        
        self.model_name = model_name
        self.device = device
        self.embedding_dim = 512  # ArcFace standard
        
        # Initialize FaceAnalysis (downloads models on first run)
        print(f"🔥 Initializing ArcFace encoder with {model_name} on {device}...")
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            **kwargs
        )
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=det_size)
        
        print(f"✅ ArcFace encoder ready! Embedding dim: {self.embedding_dim}D")
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 512D ArcFace embedding from face image.
        
        This is the main interface method that matches SemanticEncoder protocol.
        Returns just the embedding vector for compatibility.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            512D normalized embedding vector
        
        Raises:
            ValueError: If no face detected in image
        """
        # Detect and extract face embeddings
        faces = self.app.get(image)
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Use largest face if multiple detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            print(f"ℹ️ Multiple faces detected ({len(faces)}), using largest")
        
        face = faces[0]
        
        # Get normalized embedding (InsightFace already L2-normalizes)
        return face.normed_embedding  # Shape: (512,)
    
    def encode_with_info(
        self,
        image: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract embedding with additional face detection info.
        
        Use this when you need bbox, landmarks, age, gender, etc.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Tuple of (embedding, face_info_dict)
        
        Raises:
            ValueError: If no face detected in image
        """
        # Detect and extract face embeddings
        faces = self.app.get(image)
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        # Use largest face if multiple detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            print(f"ℹ️ Multiple faces detected ({len(faces)}), using largest")
        
        face = faces[0]
        
        face_info = {
            'bbox': face.bbox.tolist(),
            'det_score': float(face.det_score),
            'landmark_2d_106': face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else None,
            'pose': face.pose.tolist() if hasattr(face, 'pose') else None,
            'age': face.age if hasattr(face, 'age') else None,
            'gender': face.gender if hasattr(face, 'gender') else None,
        }
        
        return face.normed_embedding, face_info
    
    def get_dimension(self) -> int:
        """Get embedding dimension (SemanticEncoder interface)."""
        return self.embedding_dim
    
    def encode_batch(
        self,
        images: list[np.ndarray],
        return_face_info: bool = False
    ) -> list:
        """
        Batch encode multiple images.
        
        Args:
            images: List of RGB images
            return_face_info: If True, return (embedding, face_info) tuples
        
        Returns:
            List of embeddings (or (embedding, info) tuples if return_face_info=True)
        """
        results = []
        for img in images:
            try:
                if return_face_info:
                    result = self.encode_with_info(img)
                else:
                    result = self.encode(img)
                results.append(result)
            except ValueError as e:
                # No face detected
                if return_face_info:
                    results.append((None, None))
                else:
                    results.append(None)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get encoder model information."""
        return {
            'encoder_type': 'arcface',
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'framework': 'insightface',
            'normalization': 'l2',
        }
    
    def __repr__(self) -> str:
        return (
            f"ArcFaceEncoder(model={self.model_name}, "
            f"dim={self.embedding_dim}, device={self.device})"
        )


# Convenience function for quick encoding
def encode_face_arcface(
    image: np.ndarray,
    model_name: str = "buffalo_l",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Quick utility to encode a single face with ArcFace.
    
    Args:
        image: RGB image array
        model_name: InsightFace model pack
        device: Compute device
    
    Returns:
        512D normalized embedding
        
    Raises:
        ValueError: If no face detected
    """
    encoder = ArcFaceEncoder(model_name=model_name, device=device)
    return encoder.encode(image)
