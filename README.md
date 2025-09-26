# Luna Face Recognition

A modern, PyTorch-based face recognition framework with GPU acceleration support for Windows.

## Features

- **GPU Acceleration**: Full PyTorch CUDA support for Windows
- **Multiple Models**: Support for ArcFace, FaceNet, VGGFace, and more
- **Facial Analysis**: Age, gender, race, and emotion detection
- **High Performance**: Optimized for modern hardware
- **Easy Integration**: Simple API for face recognition tasks
- **REST API**: Built-in Flask API for web integration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Luna_face_recog.git
cd Luna_face_recog

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Python API

```python
from luna_face_recog import LunaFace
import numpy as np

# Initialize with default ArcFace model
face_recognizer = LunaFace()

# Verify if two faces belong to the same person
result = face_recognizer.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']}")
print(f"Similarity: {result['similarity']:.3f}")

# Analyze facial attributes
analysis = face_recognizer.analyze("face.jpg")
print(f"Age: {analysis['age']}")
print(f"Gender: {analysis['gender']}")
print(f"Emotion: {analysis['emotion']}")

# Extract face embedding
embedding = face_recognizer.represent("face.jpg")
print(f"Embedding shape: {embedding.shape}")

# Find similar faces in a database
matches = face_recognizer.find("query.jpg", "database_folder/")
for match in matches[:5]:  # Top 5 matches
    print(f"{match['identity']}: {match['similarity']:.3f}")
```

### REST API

Start the API server:

```bash
python run_api.py
```

The API will be available at `http://localhost:5000`

#### API Endpoints

- `GET /health` - Health check
- `POST /verify` - Verify two faces
- `POST /analyze` - Analyze facial attributes
- `POST /represent` - Extract face embedding

#### Example API Usage

```python
import requests
import base64

# Encode image to base64
with open("face.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Verify faces
response = requests.post("http://localhost:5000/verify",
    json={"img1": img_base64, "img2": img_base64})
result = response.json()
print(result)
```

## Models Supported

- **ArcFace**: High accuracy face recognition (default)
- **FaceNet**: Google's face embedding model
- **VGGFace**: Oxford's VGG face model
- **AuraFace**: Advanced ArcFace variant (ONNX)

## Configuration

```python
# Initialize with specific model and device
face_recognizer = LunaFace(
    model_name="facenet",  # or "arcface", "vggface", "aurafce"
    device="cuda"  # or "cpu", "auto"
)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (optional, for acceleration)
- OpenCV
- Flask (for API)
- ONNX Runtime (for AuraFace model)

## Project Structure

```
luna_face_recog/
├── luna_face.py          # Main LunaFace class
├── models/               # Face recognition models
│   ├── face_recognition.py
│   └── facial_analysis.py
├── commons/              # Utility modules
│   ├── logger.py
│   └── image_utils.py
└── api/                  # REST API
    └── app.py
```

## Development

Run tests:

```bash
python test_luna.py
```

## License

MIT License