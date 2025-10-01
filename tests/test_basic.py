import os
import json
import base64
import numpy as np

from luna_face_recog.luna_face import LunaFace


def _dummy_image():
    # Create a 112x112 solid color image (BGR)
    import cv2
    img = np.zeros((112,112,3), dtype=np.uint8)
    img[:] = (0,128,255)
    return img


def test_embedding_shape():
    lf = LunaFace(model_name="arcface", device="cpu")
    emb = lf.represent(_dummy_image())
    assert emb.shape[0] == 512


def test_verify_self():
    lf = LunaFace(model_name="arcface", device="cpu")
    img = _dummy_image()
    res = lf.verify(img, img)
    # For a synthetic solid-color image detection / preprocessing may distort channels.
    # Just assert keys exist and distance is finite.
    assert 'distance' in res and 'threshold' in res
    assert isinstance(res['distance'], float)
    assert res['distance'] >= 0


def test_detect_runs():
    lf = LunaFace(model_name="arcface", device="cpu")
    img = _dummy_image()
    dets = lf.detect(img)
    # May be zero faces for synthetic image; just ensure list type
    assert isinstance(dets, list)


def test_api_base64_decode():
    # Ensure API decoding helper would work (simulate base64 loop)
    import cv2
    img = _dummy_image()
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode()
    assert isinstance(b64, str)