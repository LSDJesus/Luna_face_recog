#!/usr/bin/env python3
"""
VAPOR-FACE MVP Launcher

Simple launcher script for the VAPOR-FACE Minimum Viable Proof-of-Concept.
"""

import sys
import os
from pathlib import Path

# Add the vapor_face_mvp directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Simple test of core components first
    print("VAPOR-FACE MVP - Testing Core Components...")
    
    from core.face_extractor import FaceExtractor
    from core.semantic_processor import SemanticProcessor
    from core.vector_store import VectorStore
    from core.surgical_pruner_fixed import SurgicalPruner
    
    print("✓ Core components imported successfully")
    
    # Test basic functionality
    print("\\nTesting basic functionality...")
    
    # Test semantic processor
    processor = SemanticProcessor(encoder_type="mock", dimension=1024)
    print("✓ Semantic processor initialized")
    
    # Test surgical pruner
    pruner = SurgicalPruner()
    print("✓ Surgical pruner initialized")
    
    # Test vector store
    store = VectorStore(db_path="test_vapor_face.db")
    print("✓ Vector store initialized")
    
    # Test face extractor
    extractor = FaceExtractor()
    extractor_status = extractor.get_status()
    print(f"✓ Face extractor initialized ({extractor_status['detector_type']})")
    
    print("\\n" + "="*50)
    print("VAPOR-FACE MVP - Core Components Ready!")
    print("="*50)
    print()
    print("Available functionality:")
    print("1. Face detection and extraction")
    print("2. Mock semantic embedding generation")
    print("3. Surgical vector pruning with multiple strategies")
    print("4. SQLite-based vector storage")
    print("5. Systematic axis scanning")
    print()
    print("To run the GUI (when ready):")
    print("  python main.py")
    print()
    print("To run command-line tests:")
    print("  python test_core.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\\nPlease ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"❌ Initialization error: {e}")
    sys.exit(1)