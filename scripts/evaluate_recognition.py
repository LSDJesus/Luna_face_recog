import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luna_face_recog.detection.face_detector import FaceDetector  # type: ignore
from luna_face_recog.models.face_recognition import FaceRecognitionModel  # type: ignore


def load_images(folder: str) -> List[Tuple[str, np.ndarray]]:
    exts = ('.jpg', '.jpeg', '.png')
    paths = [p for p in glob.glob(os.path.join(folder, '*')) if p.lower().endswith(exts)]
    images = []
    for p in sorted(paths):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append((p, rgb))
    return images


def face_crop(detector: FaceDetector, img: np.ndarray) -> np.ndarray:
    faces = detector.detect_faces(img)
    if not faces:
        return img
    # largest face
    areas = []
    for f in faces:
        x1,y1,x2,y2 = map(int, f['bbox'])
        areas.append(max(0,(x2-x1))*max(0,(y2-y1)))
    idx = int(np.argmax(areas))
    x1,y1,x2,y2 = map(int, faces[idx]['bbox'])
    h,w = img.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(w,x2), min(h,y2)
    if x2<=x1 or y2<=y1:
        return img
    return img[y1:y2, x1:x2]


def compute_embeddings(detector, recognizer, images: List[Tuple[str,np.ndarray]], batch: int = 16):
    paths = []
    crops = []
    for p,img in images:
        crop = face_crop(detector, img)
        crop_resized = cv2.resize(crop, recognizer.input_shape)
        crops.append(crop_resized)
        paths.append(p)
    # batch process
    embeddings = []
    for i in range(0, len(crops), batch):
        batch_imgs = crops[i:i+batch]
        emb = recognizer.get_embeddings(batch_imgs)
        embeddings.append(emb)
    if embeddings:
        embs = np.vstack(embeddings)
    else:
        embs = np.empty((0,512), dtype=np.float32)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = embs / norms
    return paths, embs


def pairwise_l2(embs: np.ndarray):
    # Efficient L2 distance matrix for normalized vectors: d^2 = 2 - 2*cos
    # We'll compute cosine matrix then convert.
    sim = embs @ embs.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim))
    return dist, sim


def stats_for_tammy(dist_mat: np.ndarray):
    # exclude diagonal
    n = dist_mat.shape[0]
    if n < 2:
        return None
    tri = dist_mat[np.triu_indices(n, k=1)]
    return {
        'count_pairs': tri.size,
        'min': float(np.min(tri)),
        'max': float(np.max(tri)),
        'mean': float(np.mean(tri)),
        'median': float(np.median(tri)),
        'p90': float(np.percentile(tri,90)),
        'p95': float(np.percentile(tri,95)),
    }


def simple_cluster(paths: List[str], embs: np.ndarray, threshold: float) -> List[List[int]]:
    clusters: List[List[int]] = []
    for i in range(len(paths)):
        assigned = False
        for c in clusters:
            # compare to any member (single-link)
            for idx in c:
                d = np.linalg.norm(embs[i] - embs[idx])
                if d <= threshold:
                    c.append(i)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            clusters.append([i])
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tammy', default='Test_images/Tammy', help='Directory of Tammy images (single identity)')
    ap.add_argument('--all', default='Test_images', help='Directory of the larger image set')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--cluster-threshold', type=float, default=None, help='Optional manual L2 distance threshold for clustering')
    ap.add_argument('--max-cluster-report', type=int, default=8, help='Max clusters to list with sample filenames')
    args = ap.parse_args()

    print('Loading models...')
    detector = FaceDetector(device=args.device)
    recognizer = FaceRecognitionModel(model_name='arcface', device=args.device if args.device.startswith('cuda') else 'cpu')

    print('Loading Tammy images...')
    tammy_imgs = load_images(args.tammy)
    t_paths, t_embs = compute_embeddings(detector, recognizer, tammy_imgs)
    t_dist, t_sim = pairwise_l2(t_embs)
    t_stats = stats_for_tammy(t_dist)
    print('Tammy intra-class distance stats:')
    print(t_stats)

    # Suggest threshold: slightly above Tammy max distance
    if t_stats:
        suggested = t_stats['max'] * 1.15
    else:
        suggested = 0.9
    if args.cluster_threshold is None:
        args.cluster_threshold = suggested
    print(f'Using clustering L2 threshold: {args.cluster_threshold:.4f} (suggested {suggested:.4f})')

    # Evaluate identification for Tammy: each image should find nearest neighbor (excluding itself) below threshold
    correct_nn = 0
    for i in range(len(t_paths)):
        # exclude self
        d_row = t_dist[i].copy()
        d_row[i] = 1e9
        nn = int(np.argmin(d_row))
        if d_row[nn] <= args.cluster_threshold:
            correct_nn += 1
    if t_paths:
        print(f'Tammy top-1 self-consistency: {correct_nn}/{len(t_paths)} ({100.0*correct_nn/len(t_paths):.1f}%)')

    # Full set
    print('Loading full image set...')
    all_imgs = load_images(args.all)
    # Remove Tammy duplicates
    tammy_set = set(p for p,_ in tammy_imgs)
    all_imgs_nodup = [x for x in all_imgs if x[0] not in tammy_set]
    a_paths, a_embs = compute_embeddings(detector, recognizer, all_imgs_nodup)
    if a_embs.shape[0]:
        clusters = simple_cluster(a_paths, a_embs, args.cluster_threshold)
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        print(f'Clusters formed (excluding Tammy): {len(clusters)} | size distribution (top 15): {cluster_sizes[:15]}')
        # Report sample clusters with 2-4 size
        small_clusters = [c for c in clusters if 2 <= len(c) <= 4]
        print(f'Small clusters (2-4 images): {len(small_clusters)}')
        for c in small_clusters[:args.max_cluster_report]:
            print('Cluster size', len(c), 'samples:', [a_paths[i] for i in c])
    else:
        print('No additional images processed for clustering.')

    # Optionally write summary JSON (skip for now)

if __name__ == '__main__':
    main()
