"""Test script for Qwen2.5-VL encoder integration

Tests whether the real MMProj models can be loaded and used.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_qwen_encoder():
    """Test the Qwen2.5-VL encoder"""
    
    print("🧪 Testing Qwen2.5-VL Encoder...")
    
    try:
        from core.qwen_encoder import QwenVLEncoder, LLAMA_CPP_AVAILABLE
        
        print(f"✓ llama-cpp-python available: {LLAMA_CPP_AVAILABLE}")
        
        if not LLAMA_CPP_AVAILABLE:
            print("❌ llama-cpp-python not installed")
            print("Install with: pip install llama-cpp-python")
            return False
        
        # Check if model files exist
        models_dir = Path(__file__).parent / "Models"
        main_model = models_dir / "Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated.i1-Q6_K.gguf"
        mmproj_model = models_dir / "Qwen2.5-VL-7B-NSFW-Caption-V3.mmproj-f16.gguf"
        
        print(f"📁 Models directory: {models_dir}")
        print(f"🔍 Main model exists: {main_model.exists()}")
        print(f"🔍 MMProj model exists: {mmproj_model.exists()}")
        
        if not main_model.exists() or not mmproj_model.exists():
            print("❌ Model files not found in Models directory")
            return False
        
        # Try to initialize encoder
        print("🚀 Initializing Qwen2.5-VL encoder...")
        encoder = QwenVLEncoder(
            model_path=str(main_model),
            mmproj_path=str(mmproj_model),
            n_gpu_layers=0  # Start with CPU for testing
        )
        
        print("✓ Encoder initialized successfully!")
        print(f"📊 Model info: {encoder.get_model_info()}")
        
        # Test with a dummy image
        print("🖼️ Testing with dummy image...")
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        embedding = encoder.encode(dummy_image)
        print(f"✓ Generated embedding: {embedding.shape}")
        print(f"📈 Embedding stats: mean={np.mean(embedding):.4f}, std={np.std(embedding):.4f}")
        
        # Test semantic axes
        axes = encoder.semantic_axes
        print(f"🎯 Semantic axes available: {len(axes)}")
        for name, slice_obj in list(axes.items())[:5]:  # Show first 5
            print(f"  {name}: {slice_obj}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qwen encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_processor_auto():
    """Test semantic processor with auto encoder selection"""
    
    print("\n🧪 Testing Semantic Processor (auto mode)...")
    
    try:
        from core.semantic_processor import SemanticProcessor
        
        # Test auto mode (should try Qwen first, fall back to mock)
        processor = SemanticProcessor(encoder_type="auto")
        
        print(f"✓ Processor initialized with: {processor.encoder.__class__.__name__}")
        
        # Test processing
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = processor.process_image(dummy_image)
        
        if result["success"]:
            print("✓ Image processing successful")
            print(f"📊 Embedding dimension: {result['dimension']}")
            print(f"🎯 Semantic axes: {len(result.get('semantic_axes', {}))}")
            
            # Show some stats
            stats = result['statistics']
            print(f"📈 Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            
        else:
            print(f"❌ Processing failed: {result.get('error')}")
        
        return result["success"]
        
    except Exception as e:
        print(f"❌ Semantic processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("=" * 60)
    print("VAPOR-FACE MVP - Qwen2.5-VL Integration Test")
    print("=" * 60)
    
    # Test 1: Direct Qwen encoder
    qwen_success = test_qwen_encoder()
    
    # Test 2: Semantic processor with auto mode
    processor_success = test_semantic_processor_auto()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Qwen Encoder: {'✓ PASS' if qwen_success else '❌ FAIL'}")
    print(f"Auto Processor: {'✓ PASS' if processor_success else '❌ FAIL'}")
    
    if qwen_success and processor_success:
        print("\n🎉 All tests passed! Qwen2.5-VL integration successful!")
        print("\nNext steps:")
        print("- Run full MVP test: python test_core.py")
        print("- Test with real face images")
        print("- Compare semantic axes between mock and real encoder")
    else:
        print("\n⚠️ Some tests failed. Check requirements and model files.")
        print("\nTroubleshooting:")
        print("- Install llama-cpp-python: pip install llama-cpp-python")
        print("- Verify model files are in Models/ directory")
        print("- Check GPU/CPU compatibility")

if __name__ == "__main__":
    main()