#!/usr/bin/env python3
"""
Simple test script for Luna Face Recognition
"""

import numpy as np
from luna_face_recog import LunaFace

def test_basic_functionality():
    """Test basic LunaFace functionality"""
    print("Testing Luna Face Recognition...")

    # Initialize LunaFace
    luna = LunaFace(model_name="arcface", device="cpu")
    print(f"✓ LunaFace initialized with {luna.model_name} on {luna.device}")

    # Test with dummy data (since we don't have real images)
    dummy_img = np.random.rand(112, 112, 3).astype(np.uint8)

    try:
        # Test represent (embedding extraction)
        embedding = luna.represent(dummy_img)
        print(f"✓ Embedding extracted: shape {embedding.shape}")

        # Test analyze (facial analysis)
        analysis = luna.analyze(dummy_img)
        print(f"✓ Facial analysis: {analysis}")

        # Test verify (face verification)
        verify_result = luna.verify(dummy_img, dummy_img)
        print(f"✓ Face verification: {verify_result}")

        print("\n🎉 All basic tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    test_basic_functionality()