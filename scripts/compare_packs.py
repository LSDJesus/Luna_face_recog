"""Compare InsightFace model packs (e.g. buffalo_l vs antelopev2) on a set of images.

Purpose: Help determine whether demographic inconsistencies (age / gender) or
face detection differences are due to the underlying InsightFace model packs
or our integration layer.

Metrics collected per pack:
  - detection_rate: proportion of images with >=1 face
  - avg_faces_per_image
  - age_mean / age_std (for largest face per image when age available)
  - gender_distribution counts
  - avg_detection_time_ms

Cross-pack metrics (only if exactly 2 packs supplied):
  - single_face_overlap_count: images where both packs detect exactly one face
  - gender_agreement_rate
  - mean_abs_age_diff (for images with both ages present)

Optional per-image breakdown (largest face only) can be included with --per-image.

Usage:
  python scripts/compare_packs.py --images tests/dataset --packs buffalo_l antelopev2 --device auto --output pack_comparison.json

"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Dict, Any

import cv2

from luna_face_recog.detection.face_detector import FaceDetector
from luna_face_recog.commons.logger import Logger

logger = Logger()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(path: Path) -> List[Path]:
    if path.is_dir():
        imgs = [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        imgs.sort()
        return imgs
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    raise SystemExit(f"No images found under {path}")


def build_detectors(packs: List[str], device: str, root: str | None) -> Dict[str, FaceDetector]:
    detectors: Dict[str, FaceDetector] = {}
    for pk in packs:
        logger.info(f"Loading detector pack '{pk}'")
        detectors[pk] = FaceDetector(device=device, model_pack=pk, model_root=root)
    return detectors


def largest_face(faces: List[Dict[str, Any]]):
    if not faces:
        return None
    def area(f):
        b = f['bbox']
        return max(0, (b[2]-b[0]) * (b[3]-b[1]))
    return max(faces, key=area)


def summarize_pack(results: List[Dict[str, Any]], pack: str) -> Dict[str, Any]:
    # results list entries structure: { 'image': str, 'faces': int, 'age': float|None, 'gender': str|None, 'time_ms': float }
    n = len(results)
    faces_counts = [r['faces'] for r in results]
    detection_rate = sum(1 for r in faces_counts if r > 0) / n if n else 0.0
    avg_faces = mean(faces_counts) if faces_counts else 0.0
    ages = [r['age'] for r in results if r['age'] is not None]
    genders = [r['gender'] for r in results if r['gender']]
    gender_dist: Dict[str, int] = {}
    for g in genders:
        gender_dist[g] = gender_dist.get(g, 0) + 1
    times = [r['time_ms'] for r in results]
    return {
        'pack': pack,
        'images': n,
        'detection_rate': detection_rate,
        'avg_faces_per_image': avg_faces,
        'age_count': len(ages),
        'age_mean': mean(ages) if ages else None,
        'age_std': pstdev(ages) if len(ages) > 1 else None,
        'gender_count': len(genders),
        'gender_distribution': gender_dist,
        'avg_detection_time_ms': mean(times) if times else None,
    }


def cross_pack_analysis(per_pack: Dict[str, List[Dict[str, Any]]], packs: List[str]) -> Dict[str, Any]:
    if len(packs) != 2:
        return {}
    p1, p2 = packs
    r1 = {r['image']: r for r in per_pack[p1]}
    r2 = {r['image']: r for r in per_pack[p2]}
    common = sorted(set(r1.keys()) & set(r2.keys()))
    single_overlap = 0
    gender_agree = 0
    gender_compared = 0
    age_diffs = []
    for img in common:
        a = r1[img]
        b = r2[img]
        if a['faces'] == 1 and b['faces'] == 1:
            single_overlap += 1
            ga, gb = a.get('gender'), b.get('gender')
            if ga and gb:
                gender_compared += 1
                if ga == gb:
                    gender_agree += 1
            aa, ab = a.get('age'), b.get('age')
            if isinstance(aa, (int, float)) and isinstance(ab, (int, float)):
                age_diffs.append(abs(aa - ab))
    return {
        'single_face_overlap_count': single_overlap,
        'gender_agreement_rate': (gender_agree / gender_compared) if gender_compared else None,
        'mean_abs_age_diff': mean(age_diffs) if age_diffs else None,
        'age_diff_samples': len(age_diffs),
    }


def run(args):
    images = iter_images(Path(args.images))
    if not images:
        raise SystemExit("No images found")

    detectors = build_detectors(args.packs, args.device, args.pack_root)

    per_pack_records: Dict[str, List[Dict[str, Any]]] = {pk: [] for pk in args.packs}

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warn(f"Failed to read image {img_path}")
            continue
        # NOTE: Keeping OpenCV BGR as InsightFace expects BGR internally
        for pk, det in detectors.items():
            t0 = time.time()
            faces = det.detect_faces(img)
            dt = (time.time() - t0) * 1000.0
            lf = largest_face(faces)
            rec = {
                'image': str(img_path),
                'faces': len(faces),
                'age': lf.get('age') if lf else None,
                'gender': lf.get('gender') if lf else None,
                'time_ms': dt,
            }
            per_pack_records[pk].append(rec)

    summaries = [summarize_pack(per_pack_records[pk], pk) for pk in args.packs]
    cross = cross_pack_analysis(per_pack_records, args.packs)

    output = {
        'packs': args.packs,
        'device': args.device,
        'image_count': len(images),
        'summaries': summaries,
        'cross_pack': cross,
    }

    if args.per_image:
        output['per_image'] = per_pack_records

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Wrote comparison summary to {args.output}")
    else:
        print(json.dumps(output, indent=2))


def build_argparser():
    p = argparse.ArgumentParser(description="Compare InsightFace model packs on a set of images")
    p.add_argument('--images', required=True, help='Image folder or single image path')
    p.add_argument('--packs', nargs='+', default=['buffalo_l', 'antelopev2'], help='Model packs to compare')
    p.add_argument('--device', default='auto', help='Device (cpu|cuda|auto)')
    p.add_argument('--pack-root', default=None, help='Optional custom model root directory')
    p.add_argument('--output', default=None, help='Write JSON summary to file (otherwise prints)')
    p.add_argument('--per-image', action='store_true', help='Include per-image details in output JSON')
    return p


if __name__ == '__main__':  # pragma: no cover
    args = build_argparser().parse_args()
    run(args)
