"""
Luna Face Recognition API
"""

from flask import Flask, request, jsonify
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import traceback

from luna_face_recog import LunaFace

app = Flask(__name__)

# Initialize LunaFace
luna = LunaFace(model_name="arcface", device="cpu")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": luna.model_name, "device": luna.device})

@app.route('/verify', methods=['POST'])
def verify_faces():
    """Verify if two faces belong to the same person"""
    try:
        data = request.get_json()

        if 'img1' not in data or 'img2' not in data:
            return jsonify({"error": "Missing img1 or img2"}), 400

        # Decode base64 images
        img1_data = base64.b64decode(data['img1'])
        img2_data = base64.b64decode(data['img2'])

        img1 = Image.open(BytesIO(img1_data))
        img2 = Image.open(BytesIO(img2_data))

        # Convert to numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        # Verify faces
        result = luna.verify(img1_array, img2_array)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/analyze', methods=['POST'])
def analyze_face():
    """Analyze facial attributes"""
    try:
        data = request.get_json()

        if 'img' not in data:
            return jsonify({"error": "Missing img"}), 400

        # Decode base64 image
        img_data = base64.b64decode(data['img'])
        img = Image.open(BytesIO(img_data))
        img_array = np.array(img)

        # Analyze face
        result = luna.analyze(img_array)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/represent', methods=['POST'])
def get_embedding():
    """Extract face embedding"""
    try:
        data = request.get_json()

        if 'img' not in data:
            return jsonify({"error": "Missing img"}), 400

        # Decode base64 image
        img_data = base64.b64decode(data['img'])
        img = Image.open(BytesIO(img_data))
        img_array = np.array(img)

        # Get embedding
        embedding = luna.represent(img_array)

        return jsonify({"embedding": embedding.tolist()})

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)