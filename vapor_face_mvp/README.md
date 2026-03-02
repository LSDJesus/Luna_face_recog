# VAPOR-FACE MVP

This module powers the "Semantic Archaeology" experiments over face recognition embeddings.

## What’s new

- ArcFace encoder (InsightFace buffalo_l) for real 512D identity embeddings: `core/arcface_encoder.py`
- Linear probe training (TCAV-style) to discover semantic axes: `core/semantic_probes.py`
- Vector-space surgery helpers: `core/semantic_ops.py`
- End-to-end sweep experiment (auto-label Male/Young from InsightFace): `experiments/run_semantic_sweep.py`

## Quickstart

1) Activate your environment

```pwsh
& .venv/Scripts/Activate.ps1
```

2) Sanity check ArcFace extraction on a sample image

```pwsh
$env:PYTHONIOENCODING="utf-8"; .venv/Scripts/python.exe -c "from vapor_face_mvp.core.arcface_encoder import ArcFaceEncoder; import cv2, numpy as np; enc=ArcFaceEncoder(); img=cv2.imread(r'd:\AI\Github_Desktop\Luna_face_recog\Test_images\AI_Generated\10female.jpg'); img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB); emb=enc.encode(img); print(f'ArcFace OK: {emb.shape[0]}D, L2={np.linalg.norm(emb):.6f}')"
```

3) Run the semantic sweep (auto-labels Male/Young via InsightFace age/gender)

```pwsh
.venv/Scripts/python.exe vapor_face_mvp/experiments/run_semantic_sweep.py --root Test_images/AI_Generated --attrs Male Young --alphas 0 0.05 0.1 0.2 0.3
```

You’ll see, for each attribute, the mean change in projected score (Δscore) versus the cosine similarity to the original embedding (identity preservation). Small alphas (0.05–0.20) should produce clear, monotonic score shifts while keeping cosine > 0.97 on average.

## Train custom probes from labels (optional)

If you have your own labels (e.g., CelebA), use `SemanticProbeTrainer` to train and save probes:

```python
from vapor_face_mvp.core.semantic_probes import SemanticProbeTrainer
trainer = SemanticProbeTrainer(max_iter=1000)
results = trainer.train_all_probes(embeddings, attributes)  # embeddings: (N,512), attributes: Dict[str, (N,)]
axes = trainer.get_semantic_axes()
trainer.save_probes('probes/celeba_probes.json')
```

## Apply semantic surgery

```python
from vapor_face_mvp.core.semantic_ops import apply_semantic_shift
e_shifted = apply_semantic_shift(e, axes['Smiling'], alpha=0.2, mode='add')
```

## Notes

- ArcFace comes from InsightFace model pack `buffalo_l` and returns L2-normalized 512D vectors.
- Auto-labels use InsightFace’s age/gender head; for richer attributes, provide your own labels.
- Keep alpha small to preserve identity; validate with cosine or verification metrics.

# VAPOR-FACE MVP: Minimum Viable Proof-of-Concept

A streamlined implementation focused on semantic facial recognition research and surgical vector pruning experiments.

## Project Overview

This MVP implements the core concepts from the VAPOR-FACE project specification, specifically:

- **Document 4.2.C**: "Subtractive Semantics" - Surgical pruning of semantic vector components
- **Semantic Archaeology**: Systematic analysis of semantic axes in facial embeddings
- **Vector Database**: Storage and retrieval of embeddings and experimental results

## Architecture

```
vapor_face_mvp/
├── core/                    # Core processing components
│   ├── face_extractor.py   # Face detection and cropping (Luna Face integration)
│   ├── semantic_processor.py # Semantic embedding generation (Mock MMProj)
│   ├── surgical_pruner.py  # Vector pruning and analysis tools
│   └── vector_store.py     # SQLite-based vector database
├── gui/                     # GUI components (in development)
├── experiments/             # Experimental scripts and analysis
├── data/                    # Data storage directory
├── main.py                 # GUI application (in development)
├── launch.py               # Component testing launcher
├── test_core.py            # Core functionality tests
└── requirements.txt        # Python dependencies
```

## Key Features

### 1. Semantic Processing
- **Mock MMProj Encoder**: Simulates high-dimensional semantic vectors with interpretable axes
- **Semantic Axes**: Pre-defined mappings for facial features (eye_color, nose_shape, etc.)
- **Deterministic Encoding**: Reproducible embeddings based on image statistics

### 2. Surgical Vector Pruning
- **Multiple Strategies**: Zero, Gaussian noise, mean replacement
- **Systematic Scanning**: Test all semantic axes individually
- **Impact Analysis**: L2 distance, cosine similarity, relative impact metrics
- **Experiment Tracking**: Full history of pruning experiments

### 3. Vector Database
- **SQLite Storage**: Lightweight, file-based database
- **Metadata Tracking**: Model info, processing statistics, semantic axes
- **Experiment Storage**: Pruning results and impact metrics
- **Query Interface**: Retrieve embeddings and experiments by various criteria

### 4. Face Processing
- **Luna Face Integration**: Leverages existing face detection capabilities
- **Multi-Pack Support**: buffalo_l, antelopev2 InsightFace model packs
- **Graceful Fallback**: OpenCV Haar cascades if InsightFace unavailable
- **Standardized Output**: 512x512 normalized face crops

## Quick Start

### 1. Installation

```bash
# Navigate to the Luna Face Recognition project
cd Luna_face_recog

# The MVP is located in the vapor_face_mvp subdirectory
cd vapor_face_mvp

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Core Components

```bash
# Test all core functionality
python launch.py

# Run detailed functionality tests
python test_core.py
```

### 3. Expected Output

The launcher should show:
```
VAPOR-FACE MVP - Testing Core Components...
✓ Core components imported successfully
✓ Semantic processor initialized
✓ Surgical pruner initialized
✓ Vector store initialized
✓ Face extractor initialized (Luna/OpenCV)

VAPOR-FACE MVP - Core Components Ready!
```

## Core Experiments

### Semantic Archaeology (Document 4.2.C)

The core experiment validates semantic organization by:

1. **Baseline Extraction**: Generate semantic embedding from face image
2. **Surgical Pruning**: Null specific semantic axes (e.g., "eye_color")
3. **Impact Analysis**: Measure vector distance and similarity changes
4. **Systematic Scanning**: Test all available semantic axes

Example usage:
```python
from core.semantic_processor import SemanticProcessor
from core.surgical_pruner_fixed import SurgicalPruner

# Process face
processor = SemanticProcessor(encoder_type="mock")
result = processor.process_image(face_image)

# Test surgical pruning
pruner = SurgicalPruner()
pruning_result = pruner.prune_axis(
    vector=result["embedding"],
    axis_name="eye_color",
    axis_indices=result["semantic_axes"]["eye_color"],
    strategy="zero"
)

# Analyze impact
print(f"Impact: {pruning_result['impact_metrics']}")
```

### Systematic Axis Analysis

Test all semantic axes with multiple strategies:

```python
scan_results = pruner.systematic_axis_scan(
    vector=embedding,
    semantic_axes=semantic_axes,
    strategies=["zero", "gaussian", "mean"]
)
```

## Current Limitations (MVP Scope)

### Not Implemented Yet:
- **Real MMProj Integration**: Currently using mock encoder
- **VDM Pipeline**: Video diffusion for pose synthesis (Phase 2)
- **NeRF 3D Modeling**: 3D mesh generation (Phase 2) 
- **Temporal Tracking**: Identity drift over time (Phase 3)
- **Complete GUI**: Basic framework started, needs completion

### Mock vs. Real Implementation:
- **Mock Semantic Axes**: Pre-defined 20 categories with deterministic assignment
- **Real Goal**: Learn semantic axes from actual multimodal model projections
- **Validation Method**: Manual inspection of pruning effects on generated descriptions

## Data Flow

```
Input Image → Face Detection → Face Crop → Semantic Encoding → Vector Storage
                                               ↓
                                         Surgical Pruning
                                               ↓
                                         Impact Analysis
                                               ↓
                                         Experiment Storage
```

## Integration with Luna Face Recognition

The MVP leverages the existing Luna Face Recognition infrastructure:

- **Face Detection**: Uses Luna's InsightFace integration
- **Model Packs**: Supports buffalo_l and antelopev2
- **Device Handling**: GPU/CPU fallback from Luna's detector
- **Image Processing**: Luna's preprocessing pipeline

## Next Steps

### Phase 1 Completion:
1. **Complete GUI**: Finish the tkinter interface for interactive experiments
2. **Real MMProj**: Integrate actual multimodal model encoder
3. **Visualization**: Add vector comparison and impact plotting
4. **Export Tools**: Results export and analysis utilities

### Phase 2 Roadmap:
1. **VDM Integration**: Add video diffusion for pose synthesis
2. **NeRF Pipeline**: 3D mesh generation from keyframes
3. **Advanced Metrics**: Silhouette analysis, cluster quality metrics
4. **Batch Processing**: Handle multiple images efficiently

### Research Validation:
1. **Semantic Consistency**: Verify that pruned axes correlate with expected semantic changes
2. **Cross-Model Testing**: Compare different encoder architectures
3. **Quantitative Metrics**: Establish baseline performance measurements

## File Structure Details

### Core Components:

- **`face_extractor.py`**: Handles face detection using Luna Face Recognition, with fallback to OpenCV
- **`semantic_processor.py`**: Mock MMProj encoder with predefined semantic axes
- **`surgical_pruner_fixed.py`**: Vector pruning tools with multiple strategies and impact analysis
- **`vector_store.py`**: SQLite database for embeddings and experimental results

### Testing:

- **`launch.py`**: Quick component validation and status check
- **`test_core.py`**: Comprehensive functionality testing with detailed output

### Configuration:

- **`requirements.txt`**: Python dependencies including ML libraries and GUI components
- **Database**: Auto-created SQLite file for persistent storage

This MVP provides a solid foundation for semantic facial recognition research while maintaining focus on the core surgical pruning experiments outlined in the VAPOR-FACE specification.