"""
Face Detection Module for Luna Face Recognition
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
import sys
import torch

from ..commons.logger import Logger

logger = Logger()


class FaceDetector:
    """Face Detection wrapper.

    Currently uses InsightFace FaceAnalysis for detection. Attempts to run on GPU
    if the caller specifies a CUDA device and it's available; otherwise falls back
    to CPU seamlessly. Previous version forced CPU due to early RTX 5090 support
    gaps; this version retries GPU first because newer ONNX Runtime / InsightFace
    builds may support the architecture. If GPU init fails for any reason, we log
    and fall back to CPU automatically so the pipeline keeps working.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device.lower()
        self.detector = None
        self.execution_mode = "cpu"  # tracked for diagnostics
        self._load_detector()

    def _load_detector(self):
        """Load InsightFace detector with GPU-first strategy and graceful fallback."""
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            self.detector = FaceAnalysis(name='buffalo_l', root='~/.insightface')

            # Determine desired context based on requested device and availability
            want_gpu = self.device.startswith("cuda") and torch.cuda.is_available()
            attempted_gpu = False

            if want_gpu:
                try:
                    self.detector.prepare(ctx_id=0)
                    self.execution_mode = "cuda:0"
                    logger.info("InsightFace detector loaded on CUDA (ctx_id=0)")
                    return
                except Exception as gpu_e:
                    attempted_gpu = True
                    logger.warn(f"InsightFace GPU init failed ({gpu_e}); falling back to CPU")

            # CPU fallback
            self.detector.prepare(ctx_id=-1)
            self.execution_mode = "cpu"
            if attempted_gpu:
                logger.info("InsightFace detector loaded on CPU (GPU fallback)")
            else:
                logger.info("InsightFace detector loaded on CPU")

        except Exception as e:
            logger.warn(f"Could not load InsightFace: {e}, using OpenCV Haar cascades")
            # Fallback to OpenCV Haar cascades
            try:
                import cv2
                cascade_path = cv2.__file__.replace('__init__.py', 'data/haarcascades/haarcascade_frontalface_default.xml')
                self.detector = cv2.CascadeClassifier(cascade_path)
                if self.detector.empty():
                    raise Exception("Cascade classifier is empty")
                self.execution_mode = "cpu-haar"
                logger.info("Haar cascade face detector loaded (CPU)")
            except Exception as cv_e:
                logger.warn(f"OpenCV Haar cascades not available: {cv_e}")
                self.detector = None
                self.execution_mode = "unavailable"

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image

        Args:
            img: Input image (H, W, C)

        Returns:
            List of face detections with bounding boxes and landmarks
        """
        if self.detector is None:
            logger.error("No face detector available")
            return []

        try:
            # Late imports for type hints / checks
            try:
                from insightface.app import FaceAnalysis  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                FaceAnalysis = None  # type: ignore

            # Branch on actual instance types to keep static analyzers quiet
            # NOTE: The previous implementation checked `'FaceAnalysis' in globals()` which
            # failed because FaceAnalysis is imported in this function scope (locals), thus
            # always falling through and logging an unknown detector type. We simplify to a
            # direct isinstance check.
            if FaceAnalysis is not None and isinstance(self.detector, FaceAnalysis):  # type: ignore
                faces = self.detector.get(img)  # type: ignore
                return self._process_insightface_results(faces)
            elif isinstance(self.detector, cv2.CascadeClassifier):  # type: ignore
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                faces = self.detector.detectMultiScale(gray, 1.1, 4)
                return self._process_opencv_results(faces)
            else:
                logger.error(f"Unknown detector type: {type(self.detector)}")
                return []

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    def _process_insightface_results(self, faces: List) -> List[Dict[str, Any]]:
        """Process InsightFace detection results"""
        results = []
        for face in faces:
            # Face objects can be either dict-like or objects with attributes.
            try:
                if isinstance(face, dict):
                    bbox = face.get('bbox')
                    confidence = face.get('det_score', 1.0)
                    landmarks = face.get('kps')
                else:  # object from insightface
                    bbox = getattr(face, 'bbox', None)
                    confidence = getattr(face, 'det_score', 1.0)
                    landmarks = getattr(face, 'kps', None)

                if bbox is None:
                    continue

                # Ensure bbox is a python list of ints / floats length 4
                if hasattr(bbox, 'tolist'):
                    bbox_list = bbox.tolist()
                else:
                    bbox_list = list(bbox)

                # Demographic attributes (may be present already, so capture here to avoid re-running analysis)
                age_attr = getattr(face, 'age', None) if not isinstance(face, dict) else face.get('age')
                gender_attr = None
                if not isinstance(face, dict):
                    gender_attr = getattr(face, 'gender', None) or getattr(face, 'sex', None)
                else:
                    gender_attr = face.get('gender') or face.get('sex')

                # Normalize gender label if present
                gender_label = None
                gender_conf = None
                if gender_attr is not None:
                    if isinstance(gender_attr, (int, float)):
                        gender_label = 'Male' if int(gender_attr) == 1 else 'Female'
                    elif isinstance(gender_attr, str):
                        gender_label = gender_attr.capitalize()
                    elif isinstance(gender_attr, (list, tuple)) and gender_attr:
                        # Could be (label, score) or (score,)
                        if isinstance(gender_attr[0], str):
                            gender_label = gender_attr[0].capitalize()
                            if len(gender_attr) > 1 and isinstance(gender_attr[1], (int, float)):
                                gender_conf = float(gender_attr[1])
                        elif isinstance(gender_attr[0], (int, float)):
                            gender_label = 'Male' if int(gender_attr[0]) == 1 else 'Female'
                    elif isinstance(gender_attr, dict):
                        try:
                            k, v = max(gender_attr.items(), key=lambda kv: kv[1])
                            gender_label = k.capitalize()
                            gender_conf = float(v)
                        except Exception:
                            pass

                if landmarks is not None and hasattr(landmarks, 'tolist'):
                    landmarks_val = landmarks.tolist()
                else:
                    landmarks_val = landmarks

                results.append({
                    'bbox': bbox_list,
                    'confidence': float(confidence) if confidence is not None else 1.0,
                    'landmarks': landmarks_val,
                    'age': float(age_attr) if age_attr is not None else None,
                    'gender': gender_label,
                    'gender_confidence': gender_conf
                })
            except Exception as fe:  # pragma: no cover - defensive
                logger.warn(f"Failed to parse face result: {fe}")
                continue
        if not results:
            logger.debug("InsightFace returned zero faces for this image")
        return results

    def _process_opencv_results(self, faces: Any) -> List[Dict[str, Any]]:
        """Process OpenCV Haar cascade results"""
        results = []
        if faces is not None:
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.8,  # Haar cascades don't provide confidence
                    'landmarks': None
                })
        return results

    def extract_faces(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Extract face regions from image

        Args:
            img: Input image
            detections: Face detection results

        Returns:
            List of extracted face images
        """
        faces = []
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                face_img = img[y1:y2, x1:x2]
                faces.append(face_img)

        return faces