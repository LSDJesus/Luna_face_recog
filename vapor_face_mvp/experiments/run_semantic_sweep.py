"""
Run a semantic sweep experiment using ArcFace embeddings and auto-labeled attributes.

Pipeline:
1) Load images from a root directory (default: ../../Test_images)
2) Extract ArcFace embeddings + face info (age/gender)
3) Auto-label attributes: Male (gender==1), Young (age<30)
4) Train linear probes to get semantic directions
5) Sweep alpha and report attribute score shift vs cosine preservation

Usage (from repo root):
  .venv/Scripts/python.exe vapor_face_mvp/experiments/run_semantic_sweep.py \
      --root Test_images/AI_Generated --attrs Male Young
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2

import sys
PKG_ROOT = Path(__file__).resolve().parents[2]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from vapor_face_mvp.core.arcface_encoder import ArcFaceEncoder
from vapor_face_mvp.core.semantic_probes import SemanticProbeTrainer
from vapor_face_mvp.core.semantic_ops import (
    apply_semantic_shift,
    cosine_similarity,
)


def collect_dataset(root: Path) -> Tuple[List[np.ndarray], List[Dict]]:
    images: List[np.ndarray] = []
    infos: List[Dict] = []
    enc = ArcFaceEncoder()
    
    paths = sorted(list(root.rglob("*.jpg")) + list(root.rglob("*.png")))
    for p in paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            emb, info = enc.encode_with_info(img)
            images.append(emb)
            infos.append(info)
        except Exception:
            # No face or failure; skip
            continue
    return images, infos


def build_attributes(infos: List[Dict], attrs: List[str]) -> Dict[str, np.ndarray]:
    attr_data: Dict[str, List[int]] = {a: [] for a in attrs}
    for info in infos:
        # InsightFace returns gender as 0/1 and age as int
        gender = info.get("gender", None)
        age = info.get("age", None)
        for a in attrs:
            if a.lower() == "male":
                val = 1 if gender == 1 else 0
            elif a.lower() == "young":
                val = 1 if (age is not None and age < 30) else 0
            else:
                # Unknown attribute—default to 0, will be dropped if constant
                val = 0
            attr_data[a].append(val)
    # Convert to arrays
    return {k: np.array(v, dtype=np.int32) for k, v in attr_data.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[2] / "Test_images"),
                    help="Root directory of images (will scan recursively)")
    ap.add_argument("--attrs", nargs="*", default=["Male", "Young"],
                    help="Attributes to train (supports Male, Young)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.3],
                    help="Alpha sweep values")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"Scanning images under: {root}")
    embeddings, infos = collect_dataset(root)
    if len(embeddings) < 10:
        print("Not enough images with detected faces. Please point --root to a directory with more faces.")
        return

    E = np.vstack(embeddings)  # (N, 512)
    attrs = build_attributes(infos, args.attrs)

    # Drop attributes that are constant (all 0 or all 1)
    attrs = {k: v for k, v in attrs.items() if v.min() != v.max()}
    if not attrs:
        print("No varying attributes available (Male/Young constant). Try a different image set.")
        return

    print(f"Training probes for attributes: {list(attrs.keys())}")
    trainer = SemanticProbeTrainer(max_iter=1000)
    results = trainer.train_all_probes(E, attrs)
    axes = trainer.get_semantic_axes()

    # Sweep
    print("\nSemantic sweep results (mean over dataset):")
    for attr, direction in axes.items():
        print(f"\nAttribute: {attr}")
        dir_vec = direction.astype(np.float32)
        base_scores = E @ dir_vec  # projection
        for a in args.alphas:
            shifted = np.vstack([apply_semantic_shift(e, dir_vec, a, mode="add") for e in E])
            shifted_scores = shifted @ dir_vec
            deltas = shifted_scores - base_scores
            cosines = np.array([cosine_similarity(e, s) for e, s in zip(E, shifted)])
            print(
                f"  alpha={a:.2f}: Δscore: mean={deltas.mean():+.4f}, std={deltas.std():.4f}; "
                f"cosine: mean={cosines.mean():.4f}, min={cosines.min():.4f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
