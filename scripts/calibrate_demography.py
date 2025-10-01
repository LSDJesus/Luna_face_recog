import argparse, json, os, sys, csv
from pathlib import Path
import cv2
import numpy as np
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luna_face_recog.detection.face_detector import FaceDetector  # type: ignore
from luna_face_recog.models.legacy_demography import LegacyDemography  # type: ignore

"""Calibration script.
Provide a CSV with: path,age,gender
Where age is approximate integer target and gender in {Male,Female}.
We will:
 - Run legacy heads (age,genden optional race disabled) with probability output
 - Aggregate per-class statistics to build a temperature scaling parameter for gender
 - Compute linear regression mapping for expected age to target age.
Outputs a JSON calib file usable at inference time.
"""

def load_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV with columns path,age,gender')
    ap.add_argument('--images-root', default='.', help='Base directory for image paths in CSV')
    ap.add_argument('--out', default='demography_calibration.json')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    entries = load_csv(args.csv)
    if not entries:
        print('No rows found in CSV')
        return

    det = FaceDetector(device=args.device)
    legacy = LegacyDemography(device=args.device, enable_age=True, enable_gender=True, enable_race=False, share_backbone=True)

    records = []
    for e in entries:
        img_path = os.path.join(args.images_root, e['path']) if not os.path.isabs(e['path']) else e['path']
        if not os.path.exists(img_path):
            print('Missing image', img_path)
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            print('Unreadable', img_path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = det.detect_faces(rgb)
        if not faces:
            print('No face for', img_path)
            continue
        # largest face
        areas = []
        for f in faces:
            x1,y1,x2,y2 = map(int,f['bbox'])
            areas.append(max(0,(x2-x1))*max(0,(y2-y1)))
        idx = int(np.argmax(areas))
        x1,y1,x2,y2 = map(int, faces[idx]['bbox'])
        crop = rgb[y1:y2, x1:x2]
        pred = legacy.predict(crop, return_probs=True)
        records.append({
            'path': e['path'],
            'target_age': int(e['age']),
            'target_gender': e['gender'].strip().capitalize(),
            'pred_age': pred.get('age'),
            'pred_age_conf': pred.get('age_confidence'),
            'pred_gender': pred.get('gender'),
            'pred_gender_conf': pred.get('gender_confidence'),
            'gender_probs': pred.get('gender_probs'),
        })

    if not records:
        print('No usable predictions')
        return

    # Age calibration: fit a simple linear a*pred + b = target
    xs = [r['pred_age'] for r in records if r['pred_age'] is not None]
    ys = [r['target_age'] for r in records if r['pred_age'] is not None]
    if len(xs) >= 2:
        x_mean = mean(xs); y_mean = mean(ys)
        num = sum((x - x_mean)*(y - y_mean) for x,y in zip(xs,ys))
        den = sum((x - x_mean)**2 for x in xs) or 1.0
        a = num / den
        b = y_mean - a * x_mean
    else:
        a, b = 1.0, 0.0

    # Gender calibration: we treat provided gender_conf as raw max prob; compute median male/female prob to create centering
    male_probs = [r['gender_probs'][1] for r in records if r.get('gender_probs') and r['target_gender']=='Male']
    female_probs = [r['gender_probs'][0] for r in records if r.get('gender_probs') and r['target_gender']=='Female']
    def median(lst):
        return float(np.median(lst)) if lst else 0.5
    m_male = median(male_probs)
    m_female = median(female_probs)

    calib = {
        'age_linear': {'a': a, 'b': b},
        'gender_medians': {'male': m_male, 'female': m_female},
        'meta': {'count': len(records)}
    }
    with open(args.out, 'w') as f:
        json.dump(calib, f, indent=2)
    print('Wrote calibration to', args.out)

if __name__ == '__main__':
    main()
