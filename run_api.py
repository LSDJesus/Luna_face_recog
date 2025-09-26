#!/usr/bin/env python3
"""
Run Luna Face Recognition API
"""

from luna_face_recog.api import app

if __name__ == '__main__':
    print("Starting Luna Face Recognition API...")
    print("API will be available at http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /verify     - Verify two faces")
    print("  POST /analyze    - Analyze facial attributes")
    print("  POST /represent  - Extract face embedding")
    print()

    app.run(host='0.0.0.0', port=5000, debug=False)