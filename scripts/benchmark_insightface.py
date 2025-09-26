import os
import sys
import time
import glob
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Ensure we are running inside the project virtual environment (.venv) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_NAME = '.venv'
expected_python_segment = os.path.join(PROJECT_ROOT, VENV_NAME, 'Scripts')
if os.name == 'nt':
    # Windows path check
    if expected_python_segment not in sys.executable:
        print(f"[FATAL] This script must be run with the project virtual environment Python.\nCurrent: {sys.executable}\nExpected to contain: {expected_python_segment}\nActivate with: .\\{VENV_NAME}\\Scripts\\Activate.ps1")
        sys.exit(1)
else:
    # POSIX
    if VENV_NAME not in sys.prefix:
        print(f"[FATAL] This script must be run inside the {VENV_NAME} virtual environment. Activate first.")
        sys.exit(1)

import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

from luna_face_recog.detection.face_detector import FaceDetector  # type: ignore
from luna_face_recog.models.facial_analysis import FacialAnalysisModel  # type: ignore
from luna_face_recog.models.legacy_demography import LegacyDemography  # type: ignore
from luna_face_recog.models.face_recognition import FaceRecognitionModel  # type: ignore

AREA_THRESHOLD_RATIO = 0.8  # faces >= 80% of largest area will be embedded


def load_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = [p for p in glob.glob(os.path.join(folder, '*')) if p.lower().endswith(exts)]
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((p, img))
    return images


def run_batch(detector, analyzer, recognizer, images, device, legacy_demo=None, legacy_mode="off"):
    results = []
    for path, img in images:
        faces = detector.detect_faces(img)
        analyses = []
        embeddings = []
        if faces:
            # Compute largest face area
            areas = []
            for f in faces:
                x1, y1, x2, y2 = map(int, f['bbox'])
                areas.append(max(0, (x2 - x1)) * max(0, (y2 - y1)))
            max_area = max(areas) if areas else 0
            # Determine which faces qualify (>=80% largest area)
            qualifying = []
            for f, area in zip(faces, areas):
                if max_area == 0 or area >= AREA_THRESHOLD_RATIO * max_area:
                    qualifying.append(f)

            # Extract crops for qualifying faces
            extracted = detector.extract_faces(img, qualifying)
            for det, fimg in zip(qualifying, extracted):
                # If detection already has demographics, use them; else call analyzer once
                use_legacy = legacy_demo is not None and legacy_mode != 'off'
                base_det = {
                    'age': det.get('age'),
                    'gender': det.get('gender'),
                    'gender_confidence': det.get('gender_confidence'),
                }
                if use_legacy:
                    # Extra guard for static analysis: ensure object has predict
                    if legacy_demo is not None and hasattr(legacy_demo, 'predict'):
                        legacy_out = legacy_demo.predict(fimg)
                    else:  # fallback: disable legacy for this face
                        legacy_out = {}
                    if legacy_mode == 'hybrid':
                        # Prefer detector age/gender if present, else legacy
                        age = base_det['age'] if base_det['age'] is not None else legacy_out.get('age')
                        gender = base_det['gender'] if base_det['gender'] else legacy_out.get('gender')
                        gender_conf = base_det['gender_confidence'] if base_det['gender_confidence'] is not None else legacy_out.get('gender_confidence')
                        analyses.append({
                            'age': age if age is not None else 25.0,
                            'gender': gender if gender else 'Unknown',
                            'gender_confidence': gender_conf if gender_conf is not None else 0.0,
                            'race': legacy_out.get('race', 'Unknown'),
                            'race_confidence': legacy_out.get('race_confidence'),
                            'emotion': 'neutral',
                            'emotion_confidence': 0.0
                        })
                    else:  # legacy_mode == 'legacy'
                        analyses.append({
                            'age': legacy_out.get('age', 25.0),
                            'gender': legacy_out.get('gender', 'Unknown'),
                            'gender_confidence': legacy_out.get('gender_confidence', 0.0),
                            'race': legacy_out.get('race', 'Unknown'),
                            'race_confidence': legacy_out.get('race_confidence'),
                            'emotion': 'neutral',
                            'emotion_confidence': 0.0
                        })
                else:
                    if det.get('age') is not None or det.get('gender') is not None:
                        analyses.append({
                            'age': det.get('age', 25.0) if det.get('age') is not None else 25.0,
                            'gender': det.get('gender') if det.get('gender') else 'Unknown',
                            'gender_confidence': det.get('gender_confidence') if det.get('gender_confidence') is not None else 0.0,
                            'race': 'Unknown',
                            'race_confidence': 0.0,
                            'emotion': 'neutral',
                            'emotion_confidence': 0.0
                        })
                    else:
                        analyses.append(analyzer.analyze_face(fimg))

                # Prepare embedding (resize to recognizer input shape)
                face_resized = cv2.resize(fimg, recognizer.input_shape)
                emb = recognizer.get_embedding(face_resized)
                embeddings.append(emb)

        results.append({
            'path': path,
            'faces': len(faces),
            'qualified_faces': len(embeddings),
            'analysis': analyses,
            'embedding_dims': [e.shape[0] for e in embeddings]
        })
    return results


_SHARED_COMPONENTS = None


def _build_components(device, force_new=False):
    global _SHARED_COMPONENTS
    if _SHARED_COMPONENTS is not None and not force_new:
        return _SHARED_COMPONENTS
    detector = FaceDetector(device=device)
    analyzer = FacialAnalysisModel(device=device)
    recognizer = FaceRecognitionModel(model_name='arcface', device='cuda' if torch.cuda.is_available() else 'cpu')
    _SHARED_COMPONENTS = (detector, analyzer, recognizer)
    return _SHARED_COMPONENTS


def _worker_process(chunk, device, legacy_demo, legacy_mode):
    # Each worker gets its own instances (InsightFace objects are not guaranteed thread-safe)
    detector, analyzer, recognizer = _build_components(device)
    # NOTE: legacy_demo is shared read-only; predictions run on its model (GPU safe for inference).
    return run_batch(detector, analyzer, recognizer, chunk, device, legacy_demo=legacy_demo, legacy_mode=legacy_mode)


def time_run(images, device, workers=1, legacy_demo=None, legacy_mode="off"):
    start = time.perf_counter()
    if workers == 1:
        detector, analyzer, recognizer = _build_components(device)
        # Warm-up (first call often includes init overhead); exclude from timing
        if images:
            _ = run_batch(detector, analyzer, recognizer, images[:1], device)
        start = time.perf_counter()
        out = run_batch(detector, analyzer, recognizer, images, device, legacy_demo=legacy_demo, legacy_mode=legacy_mode)
        elapsed = time.perf_counter() - start
        return elapsed, out
    else:
        chunks = [[] for _ in range(workers)]
        for i, item in enumerate(images):
            chunks[i % workers].append(item)
        out = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker_process, chunk, device, legacy_demo, legacy_mode) for chunk in chunks if chunk]
            for f in as_completed(futures):
                out.extend(f.result())
    elapsed = time.perf_counter() - start
    return elapsed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tammy', default='Test_images/Tammy', help='Path to Tammy directory')
    ap.add_argument('--all', default='Test_images', help='Path to full test images directory')
    ap.add_argument('--device', default='cuda', help='cuda or cpu preference for detection/analysis')
    ap.add_argument('--workers', type=int, default=1, help='Parallel workers for full test set')
    ap.add_argument('--demography', choices=['off','legacy','hybrid'], default='off', help='Use legacy demographic heads: off|legacy|hybrid (hybrid prefers detector values when present)')
    ap.add_argument('--race', action='store_true', help='Enable race head (legacy)')
    ap.add_argument('--share-backbone', action='store_true', help='Share one backbone across legacy heads')
    args = ap.parse_args()

    print('Torch CUDA available:', torch.cuda.is_available())
    print('Selected device:', args.device)

    # Tammy batch timing
    tammy_images = load_images(args.tammy)
    print(f'Tammy set: {len(tammy_images)} images')
    legacy_demo = None
    if args.demography != 'off':
        try:
            legacy_demo = LegacyDemography(device=args.device, enable_age=True, enable_gender=True, enable_race=args.race, share_backbone=args.share_backbone)
            if not legacy_demo.available():
                print('[WARN] Legacy demography not available after init; continuing without.')
                legacy_demo = None
        except Exception as e:
            print(f'[WARN] Failed to init legacy demography: {e}')
            legacy_demo = None

    t_time, tammy_results = time_run(tammy_images, args.device, workers=1, legacy_demo=legacy_demo, legacy_mode=args.demography)
    print(f'Tammy batch elapsed: {t_time:.3f}s  ({len(tammy_images)/t_time:.2f} img/s)')

    # Full test set with workers
    all_images = load_images(args.all)
    # Remove Tammy subdir duplicates if needed
    all_images = [x for x in all_images if '/Tammy/' not in x[0] and '\\Tammy\\' not in x[0]] + tammy_images
    print(f'Full test set (including Tammy): {len(all_images)} images')
    full_time, full_results = time_run(all_images, args.device, workers=args.workers, legacy_demo=legacy_demo, legacy_mode=args.demography)
    print(f'Full set elapsed ({args.workers} workers): {full_time:.3f}s  ({len(all_images)/full_time:.2f} img/s)')

    # Simple aggregate stats
    face_counts = [r['faces'] for r in full_results]
    detected = sum(1 for c in face_counts if c > 0)
    print(f'Faces detected in {detected}/{len(face_counts)} images ({(detected/len(face_counts))*100:.1f}%)')

    # Sample demographic outputs
    demo_samples = [r for r in full_results if r['analysis']]
    print('Sample analyses:')
    for sample in demo_samples[:5]:
        print(sample['path'], sample['analysis'][0])


if __name__ == '__main__':
    main()
