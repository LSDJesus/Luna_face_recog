"""Face Extractor for VAPOR-FACE MVP

Handles face detection and cropping for semantic analysis.
Leverages Luna Face Recognition detection capabilities.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Import from parent Luna Face Recognition project
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from luna_face_recog.detection.face_detector import FaceDetector
    LUNA_AVAILABLE = True
except ImportError:
    LUNA_AVAILABLE = False
    logging.warning("Luna Face Recognition not available, using basic OpenCV detection")

logger = logging.getLogger(__name__)


class FaceExtractor:
    """Extract and normalize faces for semantic analysis"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (512, 512),
                 use_luna: bool = True,
                 detector_pack: str = "buffalo_l"):
        """
        Initialize face extractor
        
        Args:
            target_size: Output image size (width, height)
            use_luna: Whether to use Luna Face Detection (if available)
            detector_pack: InsightFace pack to use
        """
        self.target_size = target_size
        self.use_luna = use_luna and LUNA_AVAILABLE
        
        if self.use_luna:
            try:
                # Try to find Models directory
                models_path = self._find_models_path()
                self.detector = FaceDetector(
                    device="auto", 
                    model_pack=detector_pack,
                    model_root=models_path
                )
                logger.info(f"Luna Face Detector initialized with {detector_pack}")
            except Exception as e:
                logger.warning(f"Failed to initialize Luna detector: {e}, falling back to OpenCV")
                self.use_luna = False
                
        if not self.use_luna:
            # Fallback to basic OpenCV
            self._init_opencv_detector()
    
    def _find_models_path(self) -> Optional[str]:
        """Try to find the Models directory from Luna Face project"""
        current = Path(__file__).parent
        for _ in range(5):  # Search up to 5 levels up
            models_path = current / "Models"
            if models_path.exists():
                return str(models_path)
            current = current.parent
        return None
    
    def _init_opencv_detector(self):
        """Initialize basic OpenCV cascade detector"""
        try:
            self.cv_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("OpenCV Haar cascade detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV detector: {e}")
            self.cv_detector = None
    
    def extract_face(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract the primary face from an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized face crop as numpy array, or None if no face found
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.use_luna:
            return self._extract_with_luna(img_rgb)
        else:
            return self._extract_with_opencv(img_rgb)
    
    def _extract_with_luna(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face using Luna Face Detector"""
        try:
            detections = self.detector.detect_faces(img)
            if not detections:
                logger.warning("No faces detected by Luna detector")
                return None
                
            # Get the largest face (highest confidence)
            best_detection = max(detections, key=lambda x: x.get('confidence', 0))
            bbox = best_detection['bbox']
            
            return self._crop_and_normalize(img, bbox)
            
        except Exception as e:
            logger.error(f"Luna face detection failed: {e}")
            return None
    
    def _extract_with_opencv(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face using OpenCV cascade"""
        if self.cv_detector is None:
            return None
            
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.cv_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                logger.warning("No faces detected by OpenCV")
                return None
                
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            bbox = [x, y, x + w, y + h]
            
            return self._crop_and_normalize(img, bbox)
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return None
    
    def _crop_and_normalize(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop and normalize face region
        
        Args:
            img: Source image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Normalized face crop
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract face region
        face_crop = img[y1:y2, x1:x2]
        
        # Resize to target size
        face_normalized = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return face_normalized
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status information"""
        return {
            "luna_available": LUNA_AVAILABLE,
            "using_luna": self.use_luna,
            "target_size": self.target_size,
            "detector_type": "Luna" if self.use_luna else "OpenCV"
        }