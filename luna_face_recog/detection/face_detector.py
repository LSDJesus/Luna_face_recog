"""
Face Detection Module for Luna Face Recognition
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
import sys
import torch
from pathlib import Path

from ..commons.logger import Logger

logger = Logger()


class FaceDetector:
    """Face Detection (and basic age/gender) wrapper.

    Uses InsightFace `FaceAnalysis` model packs. Default pack: `buffalo_l`.
    You can try alternative packs (e.g. `antelopev2`) which ship different
    detection / recognition backbones and sometimes updated age/gender heads.

    Args:
        device: 'cpu' | 'cuda' | 'cuda:0' etc.
        model_pack: Name of InsightFace model pack (e.g. 'buffalo_l', 'antelopev2').
        model_root: Optional local directory containing model subfolders. If
            provided, pack files are expected under `<model_root>/<model_pack>`.
            Falls back to default (~/.insightface) if not set.
    """

    def __init__(self, device: str = "cpu", model_pack: str = "buffalo_l", model_root: str | None = None):
        self.device = device.lower()
        self.model_pack = model_pack
        # Normalize and expand model root so that user can pass either the parent directory
        # containing the pack folder OR the pack folder itself. InsightFace expects root/<pack>.
        raw_root = model_root or '~/.insightface'
        expanded_root = os.path.abspath(os.path.expanduser(raw_root))
        # If user passed the pack directory itself, adjust to its parent so FaceAnalysis finds it
        if os.path.basename(os.path.normpath(expanded_root)) == self.model_pack and os.path.isdir(expanded_root):
            parent = os.path.dirname(os.path.normpath(expanded_root)) or expanded_root
            if os.path.isdir(parent):
                logger.debug(f"Adjusting provided model_root '{expanded_root}' to parent '{parent}' for pack '{self.model_pack}'")
                expanded_root = parent
        self.model_root = expanded_root
        # Set INSIGHTFACE_HOME so internal helper uses the same root (avoids duplicate nested roots)
        os.environ.setdefault('INSIGHTFACE_HOME', self.model_root)
        self.detector = None
        self.execution_mode = "cpu"  # tracked for diagnostics
        self._load_detector()

    def _load_detector(self):
        """Load InsightFace detector with GPU-first strategy, pack selection, and graceful fallback."""
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            debug = bool(os.environ.get("LUNA_DEBUG_DETECTOR"))
            if debug:
                logger.info(f"[DEBUG] Attempting to load FaceAnalysis pack='{self.model_pack}' root='{self.model_root}'")

            # Pre-flight directory diagnostics
            pack_dir = os.path.join(self.model_root, self.model_pack)
            if debug:
                if os.path.isdir(pack_dir):
                    listing = os.listdir(pack_dir)
                    logger.info(f"[DEBUG] Pack directory exists: {pack_dir} -> files: {listing}")
                else:
                    logger.warn(f"[DEBUG] Pack directory does not exist yet: {pack_dir}")

            # Heuristic: if essential files already present, we expect no download.
            expected_markers = {
                'buffalo_l': ['det_10g.onnx', 'w600k_r50.onnx', 'genderage.onnx'],
                'antelopev2': ['scrfd_10g_bnkps.onnx', 'glintr100.onnx', 'genderage.onnx']
            }
            markers = expected_markers.get(self.model_pack, [])
            have_all = all(os.path.isfile(os.path.join(pack_dir, m)) for m in markers) if markers else False
            if debug:
                logger.info(f"[DEBUG] Expected markers for '{self.model_pack}': {markers} present={have_all}")

            self.detector = FaceAnalysis(name=self.model_pack, root=self.model_root)

            # Determine desired context based on requested device and availability
            want_gpu = self.device.startswith("cuda") and torch.cuda.is_available()
            attempted_gpu = False

            if want_gpu:
                try:
                    self.detector.prepare(ctx_id=0)
                    self.execution_mode = "cuda:0"
                    logger.info(f"InsightFace detector pack '{self.model_pack}' loaded on CUDA (ctx_id=0)")
                    return
                except Exception as gpu_e:
                    attempted_gpu = True
                    logger.warn(f"InsightFace GPU init failed for pack '{self.model_pack}' ({gpu_e}); falling back to CPU")

            # CPU fallback
            self.detector.prepare(ctx_id=-1)
            self.execution_mode = "cpu"
            if attempted_gpu:
                logger.info(f"InsightFace detector pack '{self.model_pack}' loaded on CPU (GPU fallback)")
            else:
                logger.info(f"InsightFace detector pack '{self.model_pack}' loaded on CPU")

        except Exception as e:
            logger.warn(f"Could not load InsightFace pack '{self.model_pack}' (root='{self.model_root}'): {e}; attempting OpenCV Haar cascades fallback")
            # Fallback to OpenCV Haar cascades
            try:
                import cv2
                cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', 'haarcascade_frontalface_default.xml')
                if not os.path.isfile(cascade_path):
                    # Alternate common location (older opencv builds)
                    alt = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
                    if os.path.isfile(alt):
                        cascade_path = alt
                if not os.path.isfile(cascade_path):
                    raise FileNotFoundError(f"Haar cascade not found at {cascade_path}")
                self.detector = cv2.CascadeClassifier(cascade_path)
                if self.detector.empty():
                    raise RuntimeError(f"Cascade classifier failed to load from {cascade_path}")
                self.execution_mode = "cpu-haar"
                logger.info("Haar cascade face detector loaded (CPU)")
            except Exception as cv_e:
                logger.warn(f"OpenCV Haar cascades not available / failed: {cv_e}")
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