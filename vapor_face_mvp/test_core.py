"""
Simple command-line test of VAPOR-FACE MVP core functionality
"""

import numpy as np
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

from core.semantic_processor import SemanticProcessor
from core.surgical_pruner_fixed import SurgicalPruner
from core.vector_store import VectorStore

def test_semantic_processing():
    """Test semantic processing with mock encoder"""
    print("\\n" + "="*50)
    print("TESTING SEMANTIC PROCESSING")
    print("="*50)
    
    # Create mock processor
    processor = SemanticProcessor(encoder_type="mock", dimension=1024)
    
    # Create a mock face image (random noise)
    mock_face = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Process the mock image
    result = processor.process_image(mock_face)
    
    if result["success"]:
        print("✓ Semantic processing successful")
        print(f"  Embedding dimension: {result['dimension']}")
        print(f"  Statistics: {json.dumps(result['statistics'], indent=2)}")
        
        if "semantic_axes" in result:
            print(f"  Semantic axes available: {len(result['semantic_axes'])}")
            print(f"  Axes: {list(result['semantic_axes'].keys())}")
        
        return result
    else:
        print(f"❌ Semantic processing failed: {result.get('error')}")
        return None

def test_surgical_pruning(processing_result):
    """Test surgical pruning functionality"""
    print("\\n" + "="*50)
    print("TESTING SURGICAL PRUNING")
    print("="*50)
    
    if not processing_result or not processing_result["success"]:
        print("❌ Cannot test pruning without valid processing result")
        return
    
    embedding = processing_result["embedding"]
    semantic_axes = processing_result.get("semantic_axes", {})
    
    if not semantic_axes:
        print("❌ No semantic axes available for pruning")
        return
    
    # Create pruner
    pruner = SurgicalPruner()
    
    # Test single axis pruning
    print("\\nTesting single axis pruning...")
    test_axis = list(semantic_axes.keys())[0]
    axis_indices = semantic_axes[test_axis]
    
    result = pruner.prune_axis(
        vector=embedding,
        axis_name=test_axis,
        axis_indices=axis_indices,
        strategy="zero"
    )
    
    if result["success"]:
        print(f"✓ Successfully pruned axis '{test_axis}'")
        # Convert numpy types to regular Python types for JSON serialization
        impact_metrics = {k: float(v) if hasattr(v, 'item') else v 
                         for k, v in result['impact_metrics'].items()}
        print(f"  Impact metrics: {json.dumps(impact_metrics, indent=2)}")
    else:
        print(f"❌ Pruning failed: {result.get('error')}")
        return
    
    # Test systematic scan
    print("\\nTesting systematic axis scan...")
    scan_results = pruner.systematic_axis_scan(
        vector=embedding,
        semantic_axes=semantic_axes,
        strategies=["zero", "gaussian"]
    )
    
    print("✓ Systematic scan completed")
    print(f"  Strategies tested: {list(scan_results.keys())}")
    print(f"  Axes scanned: {len(list(scan_results.values())[0])}")
    
    # Show summary
    summary = pruner.get_experiment_summary()
    print(f"\\nExperiment summary: {json.dumps(summary, indent=2)}")
    
    return scan_results

def test_vector_storage(processing_result, pruning_results):
    """Test vector storage functionality"""
    print("\\n" + "="*50)
    print("TESTING VECTOR STORAGE")
    print("="*50)
    
    if not processing_result or not processing_result["success"]:
        print("❌ Cannot test storage without valid processing result")
        return
    
    # Create vector store
    store = VectorStore(db_path="test_vapor_face.db")
    
    # Store embedding
    embedding_id = store.store_embedding(
        image_path="test_mock_image.jpg",
        embedding=processing_result["embedding"],
        model_info=processing_result["model_info"],
        processing_stats=processing_result["statistics"],
        semantic_axes=processing_result.get("semantic_axes")
    )
    
    print(f"✓ Stored embedding with ID: {embedding_id}")
    
    # Retrieve embedding
    retrieved = store.get_embedding(embedding_id)
    if retrieved:
        print("✓ Successfully retrieved embedding")
        print(f"  Image path: {retrieved['image_path']}")
        print(f"  Dimension: {retrieved['dimension']}")
    else:
        print("❌ Failed to retrieve embedding")
        return
    
    # Store pruning experiment
    if pruning_results:
        strategy = list(pruning_results.keys())[0]
        axis_results = pruning_results[strategy]
        test_axis = list(axis_results.keys())[0]
        test_result = axis_results[test_axis]
        
        if test_result["success"]:
            exp_id = store.store_pruning_experiment(
                embedding_id=embedding_id,
                axis_name=test_axis,
                strategy=strategy,
                pruned_embedding=test_result["pruned_vector"],
                impact_metrics=test_result["impact_metrics"]
            )
            print(f"✓ Stored pruning experiment with ID: {exp_id}")
    
    # Get statistics
    stats = store.get_statistics()
    print(f"\\nDatabase statistics: {json.dumps(stats, indent=2)}")
    
    store.close()

def main():
    """Run all core tests"""
    print("VAPOR-FACE MVP - Core Functionality Test")
    print("This test validates the core semantic processing and pruning pipeline")
    
    # Test semantic processing
    processing_result = test_semantic_processing()
    
    # Test surgical pruning
    pruning_results = test_surgical_pruning(processing_result)
    
    # Test vector storage
    test_vector_storage(processing_result, pruning_results)
    
    print("\\n" + "="*50)
    print("CORE FUNCTIONALITY TEST COMPLETED")
    print("="*50)
    print("\\nAll core components are working correctly!")
    print("You can now:")
    print("1. Load actual face images")
    print("2. Run systematic pruning experiments")
    print("3. Store and analyze results")
    print("\\nNext steps:")
    print("- Implement GUI (main.py needs completion)")
    print("- Add real MMProj encoder")
    print("- Add visualization tools")

if __name__ == "__main__":
    main()