import argparse, json, os, sys, csv, math
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luna_face_recog.detection.face_detector import FaceDetector  # type: ignore
from luna_face_recog.models.legacy_demography import LegacyDemography  # type: ignore

# CSV columns: path,age,gender

def load_rows(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def evaluate(csv_path, device, temps, images_root='.'):
    rows = load_rows(csv_path)
    det = FaceDetector(device=device)
    legacy = LegacyDemography(device=device, enable_age=False, enable_gender=True, enable_race=False, share_backbone=True)

    # Pre-extract crops to avoid recomputation per temperature
    samples = []
    for r in rows:
        p = r['path']
        img_path = os.path.join(images_root, p) if not os.path.isabs(p) else p
        bgr = cv2.imread(img_path)
        if bgr is None:
            print('[WARN] unreadable', p)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = det.detect_faces(rgb)
        if not faces:
            print('[WARN] no face', p)
            continue
        # largest face
        areas = []
        for f in faces:
            x1,y1,x2,y2 = map(int,f['bbox'])
            areas.append(max(0,(x2-x1))*max(0,(y2-y1)))
        idx = int(np.argmax(areas))
        x1,y1,x2,y2 = map(int, faces[idx]['bbox'])
        crop = rgb[y1:y2, x1:x2]
        samples.append({
            'target_gender': r['gender'].strip().capitalize(),
            'crop': crop,
            'path': p
        })

    if not samples:
        print('No usable samples')
        return []

    results = []
    for T in temps:
        correct = 0
        total = 0
        confidences = []
        margin_vals = []
        for s in samples:
            pred = legacy.predict(s['crop'], return_probs=True, gender_temperature=T)
            g = pred.get('gender')
            probs = pred.get('gender_probs')
            if isinstance(probs, list) and len(probs) == 2:
                margin = abs(probs[1] - probs[0])
            else:
                margin = None
            if g == s['target_gender']:
                correct += 1
            total += 1
            if margin is not None:
                margin_vals.append(margin)
            conf = pred.get('gender_confidence')
            if conf is not None:
                confidences.append(conf)
        acc = correct / total if total else 0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        avg_margin = float(np.mean(margin_vals)) if margin_vals else 0.0
        results.append({
            'temperature': T,
            'accuracy': acc,
            'avg_confidence': avg_conf,
            'avg_margin': avg_margin,
            'count': total
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--temps', default='1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3')
    ap.add_argument('--images-root', default='.')
    args = ap.parse_args()
    temps = [float(x) for x in args.temps.split(',') if x.strip()]
    res = evaluate(args.csv, args.device, temps, args.images_root)
    for r in res:
        print(f"T={r['temperature']:.2f} acc={r['accuracy']*100:.1f}% avg_conf={r['avg_confidence']:.3f} avg_margin={r['avg_margin']:.3f} n={r['count']}")

if __name__ == '__main__':
    main()
