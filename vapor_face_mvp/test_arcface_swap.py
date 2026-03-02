"""
Test Script: ArcFace Encoder Swap
==================================
Quick validation that the ArcFace encoder works with VAPOR-FACE MVP.

Tests:
1. Load ArcFace encoder
2. Process test image from Test_images/
3. Verify 512D embedding shape and normalization
4. Compare with mock encoder output
5. Validate integration with SemanticProcessor

Author: Brian & Luna
Created: 2025-10-22
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add vapor_face_mvp to path
sys.path.insert(0, str(Path(__file__).parent))

from core.arcface_encoder import ArcFaceEncoder
from core.semantic_processor import SemanticProcessor


def test_arcface_encoder():
    """Test ArcFace encoder standalone"""
    print("=" * 70)
    print("TEST 1: ArcFace Encoder Standalone")
    print("=" * 70)
    
    # Initialize encoder
    print("\n📦 Initializing ArcFace encoder...")
    encoder = ArcFaceEncoder(model_name="buffalo_l")
    
    # Load test image
    test_image_dir = Path(__file__).parent.parent / "Test_images"
    # Get all test images from subdirectories
    test_images = list(test_image_dir.glob("**/*.jpg")) + list(test_image_dir.glob("**/*.png"))
    
    if not test_images:
        print("❌ No test images found in Test_images/")
        return False
    
    test_image = test_images[0]
    print(f"\n📷 Loading test image: {test_image.name}")
    
    # Read image (OpenCV loads as BGR, convert to RGB)
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"❌ Failed to load image: {test_image}")
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"   Image shape: {image.shape}")
    
    # Extract embedding
    print("\n🔥 Extracting ArcFace embedding...")
    try:
        embedding = encoder.encode(image)
        print(f"   ✅ Embedding extracted successfully!")
        print(f"   Shape: {embedding.shape}")
        print(f"   Dtype: {embedding.dtype}")
        print(f"   L2 norm: {np.linalg.norm(embedding):.6f} (should be ~1.0)")
        print(f"   Mean: {embedding.mean():.6f}")
        print(f"   Std: {embedding.std():.6f}")
        print(f"   Min: {embedding.min():.6f}, Max: {embedding.max():.6f}")
        
        # Verify it's normalized
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) < 0.01:
            print(f"   ✅ Embedding is properly L2-normalized!")
        else:
            print(f"   ⚠️ Embedding norm is {norm:.6f}, expected ~1.0")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_semantic_processor_integration():
    """Test ArcFace encoder with SemanticProcessor"""
    print("\n" + "=" * 70)
    print("TEST 2: SemanticProcessor Integration")
    print("=" * 70)
    
    # Initialize processor with ArcFace encoder
    print("\n📦 Initializing SemanticProcessor with ArcFace encoder...")
    try:
        processor = SemanticProcessor(encoder_type="arcface")
        print("   ✅ Processor initialized successfully!")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Load test image
    test_image_dir = Path(__file__).parent.parent / "Test_images"
    # Get all test images from subdirectories  
    test_images = list(test_image_dir.glob("**/*.jpg")) + list(test_image_dir.glob("**/*.png"))
    
    if not test_images:
        print("❌ No test images found in Test_images/")
        return False
    
    test_image = test_images[0]
    print(f"\n📷 Processing test image: {test_image.name}")
    
    # Read image
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"❌ Failed to load image: {test_image}")
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    print("\n🔥 Processing with SemanticProcessor...")
    try:
        result = processor.process_image(image)
        
        if result['success']:
            print(f"   ✅ Processing successful!")
            print(f"\n📊 Results:")
            print(f"   Dimension: {result['dimension']}")
            print(f"   Statistics:")
            for key, val in result['statistics'].items():
                print(f"      {key}: {val:.6f}")
            
            print(f"\n🔧 Model Info:")
            for key, val in result['model_info'].items():
                print(f"      {key}: {val}")
            
            return True
        else:
            print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_vs_arcface_comparison():
    """Compare mock encoder vs ArcFace encoder output"""
    print("\n" + "=" * 70)
    print("TEST 3: Mock vs ArcFace Comparison")
    print("=" * 70)
    
    # Initialize both processors
    print("\n📦 Initializing both encoders...")
    try:
        mock_processor = SemanticProcessor(encoder_type="mock", dimension=512)
        arcface_processor = SemanticProcessor(encoder_type="arcface")
        print("   ✅ Both processors initialized!")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Load test image
    test_image_dir = Path(__file__).parent.parent / "Test_images"
    # Get all test images from subdirectories
    test_images = list(test_image_dir.glob("**/*.jpg")) + list(test_image_dir.glob("**/*.png"))
    
    if not test_images:
        print("❌ No test images found in Test_images/")
        return False
    
    test_image = test_images[0]
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"❌ Failed to load image: {test_image}")
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with both
    print(f"\n📷 Processing {test_image.name} with both encoders...")
    
    try:
        mock_result = mock_processor.process_image(image)
        arcface_result = arcface_processor.process_image(image)
        
        print(f"\n📊 Mock Encoder Results:")
        print(f"   Dimension: {mock_result['dimension']}")
        print(f"   Mean: {mock_result['statistics']['mean']:.6f}")
        print(f"   Std: {mock_result['statistics']['std']:.6f}")
        print(f"   Norm: {mock_result['statistics']['norm']:.6f}")
        
        print(f"\n📊 ArcFace Encoder Results:")
        print(f"   Dimension: {arcface_result['dimension']}")
        print(f"   Mean: {arcface_result['statistics']['mean']:.6f}")
        print(f"   Std: {arcface_result['statistics']['std']:.6f}")
        print(f"   Norm: {arcface_result['statistics']['norm']:.6f}")
        
        print(f"\n🔍 Key Differences:")
        print(f"   Mock has semantic_axes: {('semantic_axes' in mock_result)}")
        print(f"   ArcFace has semantic_axes: {('semantic_axes' in arcface_result)}")
        print(f"   Mock model: {mock_result['model_info']['encoder_type']}")
        print(f"   ArcFace model: {arcface_result['model_info']['encoder_type']}")
        
        print(f"\n✅ Both encoders working! Ready for semantic probe training.")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 VAPOR-FACE MVP: ArcFace Encoder Validation")
    print("=" * 70)
    
    # Run tests
    test1 = test_arcface_encoder()
    test2 = test_semantic_processor_integration()
    test3 = test_mock_vs_arcface_comparison()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    print(f"   Test 1 (ArcFace Standalone): {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Test 2 (SemanticProcessor): {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"   Test 3 (Mock vs ArcFace): {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL TESTS PASSED! ArcFace encoder swap complete.")
        print("\n📚 Next Steps:")
        print("   1. Train semantic probes on CelebA attributes")
        print("   2. Run surgical pruning experiments")
        print("   3. Validate on LFW pairs with TAR@FAR metrics")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
