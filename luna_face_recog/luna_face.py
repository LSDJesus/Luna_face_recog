"""
Luna Face Recognition - Main Interface
"""

import os
import numpy as np
from typing import List, Union, Dict, Any
from pathlib import Path

from .models.face_recognition import FaceRecognitionModel
from .models.facial_analysis import FacialAnalysisModel
from .detection.face_detector import FaceDetector
from .commons.image_utils import load_image, preprocess_image
from .commons.logger import Logger

logger = Logger()


class LunaFace:
    """
    Main Luna Face Recognition class
    """

    def __init__(self, model_name: str = "arcface", device: str = "auto"):
        """
        Initialize Luna Face Recognition

        Args:
            model_name: Face recognition model to use ('arcface', 'facenet', 'vggface')
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)

        # Initialize models
        self.face_recognizer = FaceRecognitionModel(model_name, self.device)
        self.facial_analyzer = FacialAnalysisModel(self.device)
        self.face_detector = FaceDetector(self.device)

        logger.info(f"LunaFace initialized with {model_name} on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def verify(self, img1_path: Union[str, np.ndarray],
               img2_path: Union[str, np.ndarray],
               model_name: Union[str, None] = None) -> Dict[str, Any]:
        """
        Verify if two faces belong to the same person

        Args:
            img1_path: Path to first image or image array
            img2_path: Path to second image or image array
            model_name: Override model name

        Returns:
            Dictionary with verification results
        """
        # Load images
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        # Preprocess
        img1_processed = preprocess_image(img1, self.face_recognizer.input_shape)
        img2_processed = preprocess_image(img2, self.face_recognizer.input_shape)

        # Get embeddings
        model = model_name or self.model_name
        if model != self.model_name:
            # Switch model if needed
            temp_recognizer = FaceRecognitionModel(model, self.device)
            embedding1 = temp_recognizer.get_embedding(img1_processed)
            embedding2 = temp_recognizer.get_embedding(img2_processed)
        else:
            embedding1 = self.face_recognizer.get_embedding(img1_processed)
            embedding2 = self.face_recognizer.get_embedding(img2_processed)

        # Calculate similarity
        distance = np.linalg.norm(embedding1 - embedding2)
        similarity = 1 / (1 + distance)  # Convert distance to similarity

        # Determine if same person (threshold-based)
        threshold = self.face_recognizer.get_threshold()
        verified = distance < threshold

        return {
            "verified": verified,
            "distance": float(distance),
            "similarity": float(similarity),
            "threshold": threshold,
            "model": model
        }

    def analyze(self, img_path: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze facial attributes

        Args:
            img_path: Path to image or image array

        Returns:
            Dictionary with analysis results
        """
        img = load_image(img_path)
        img_processed = preprocess_image(img, self.facial_analyzer.input_shape)

        # Use the facial analyzer
        results = self.facial_analyzer.analyze_face(img_processed)

        return results

    def find(self, img_path: Union[str, np.ndarray],
             db_path: str,
             model_name: Union[str, None] = None) -> List[Dict[str, Any]]:
        """
        Find matching faces in database

        Args:
            img_path: Path to query image
            db_path: Path to database directory
            model_name: Override model name

        Returns:
            List of matching results
        """
        # Load query image
        query_img = load_image(img_path)
        query_processed = preprocess_image(query_img, self.face_recognizer.input_shape)

        # Get query embedding
        model = model_name or self.model_name
        temp_recognizer = None
        if model != self.model_name:
            temp_recognizer = FaceRecognitionModel(model, self.device)
            query_embedding = temp_recognizer.get_embedding(query_processed)
        else:
            query_embedding = self.face_recognizer.get_embedding(query_processed)

        # Search database
        results = []
        db_path_obj = Path(db_path)

        for img_file in db_path_obj.glob("**/*.jpg"):
            try:
                db_img = load_image(str(img_file))
                db_processed = preprocess_image(db_img, self.face_recognizer.input_shape)

                if temp_recognizer is not None:
                    db_embedding = temp_recognizer.get_embedding(db_processed)
                else:
                    db_embedding = self.face_recognizer.get_embedding(db_processed)

                distance = np.linalg.norm(query_embedding - db_embedding)
                similarity = 1 / (1 + distance)

                results.append({
                    "identity": img_file.stem,
                    "distance": float(distance),
                    "similarity": float(similarity)
                })

            except Exception as e:
                logger.warn(f"Error processing {img_file}: {e}")

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results

    def detect(self, img_path: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect faces in image

        Args:
            img_path: Path to image or image array

        Returns:
            List of face detections
        """
        img = load_image(img_path)
        detections = self.face_detector.detect_faces(img)
        return detections

    def extract_faces(self, img_path: Union[str, np.ndarray]) -> List[np.ndarray]:
        """
        Detect and extract faces from image

        Args:
            img_path: Path to image or image array

        Returns:
            List of extracted face images
        """
        img = load_image(img_path)
        detections = self.face_detector.detect_faces(img)
        faces = self.face_detector.extract_faces(img, detections)
        return faces

    def represent(self, img_path: Union[str, np.ndarray],
                  model_name: Union[str, None] = None) -> np.ndarray:
        """
        Extract face embedding

        Args:
            img_path: Path to image or image array
            model_name: Override model name

        Returns:
            Face embedding vector
        """
        img = load_image(img_path)
        img_processed = preprocess_image(img, self.face_recognizer.input_shape)

        model = model_name or self.model_name
        if model != self.model_name:
            temp_recognizer = FaceRecognitionModel(model, self.device)
            embedding = temp_recognizer.get_embedding(img_processed)
        else:
            embedding = self.face_recognizer.get_embedding(img_processed)

        return embedding