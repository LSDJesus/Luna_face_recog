"""
Image utilities for Luna Face Recognition
"""

import cv2
import numpy as np
from typing import Union
from pathlib import Path


def load_image(image_path: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load image from path or return if already array

    Args:
        image_path: Path to image or image array

    Returns:
        Image as numpy array in BGR format
    """
    if isinstance(image_path, np.ndarray):
        return image_path

    if isinstance(image_path, str):
        image_path_obj = Path(image_path)
    else:
        raise ValueError("image_path must be string or numpy array")

    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    img = cv2.imread(str(image_path_obj))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    return img


def preprocess_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Preprocess image for model input

    Args:
        img: Input image
        target_size: Target size (height, width)

    Returns:
        Preprocessed image
    """
    # Resize image
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))

    # Convert BGR to RGB if needed (most models expect RGB)
    # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch


def save_image(img: np.ndarray, output_path: str):
    """
    Save image to file

    Args:
        img: Image array
        output_path: Output path
    """
    cv2.imwrite(output_path, img)