import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2

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


def compute_embeddings(detector, recognizer, images: List[Tuple[str,np.ndarray]], batch: int = 32):
    paths = []
    crops = []
    for p,img in images:
        crop = face_crop(detector, img)
        crop_resized = cv2.resize(crop, recognizer.input_shape)
        crops.append(crop_resized)
        paths.append(p)
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
    sim = embs @ embs.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim))
    return dist


def complete_link_clusters(dist: np.ndarray, threshold: float) -> List[List[int]]:
    n = dist.shape[0]
    clusters = [[i] for i in range(n)]
    merged = True
    while merged:
        merged = False
        # try all pairs
        best_pair = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                ci = clusters[i]
                cj = clusters[j]
                # complete link: max distance between any members <= threshold
                max_d = 0.0
                for a in ci:
                    row = dist[a]
                    for b in cj:
                        if row[b] > max_d:
                            max_d = row[b]
                            if max_d > threshold:
                                break
                    if max_d > threshold:
                        break
                if max_d <= threshold:
                    best_pair = (i,j)
                    break
            if best_pair:
                break
        if best_pair:
            i,j = best_pair
            clusters[i].extend(clusters[j])
            del clusters[j]
            merged = True
    return clusters


def dbscan_like(dist: np.ndarray, eps: float, min_samples: int) -> List[List[int]]:
    n = dist.shape[0]
    visited = [False]*n
    labels = [-1]*n
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = [idx for idx in range(n) if dist[i, idx] <= eps]
        if len(neighbors) < min_samples:
            labels[i] = -1  # noise
            continue
        labels[i] = cluster_id
        seeds = [n2 for n2 in neighbors if n2 != i]
        while seeds:
            p = seeds.pop()
            if not visited[p]:
                visited[p] = True
                p_neighbors = [idx for idx in range(n) if dist[p, idx] <= eps]
                if len(p_neighbors) >= min_samples:
                    for q in p_neighbors:
                        if q not in seeds:
                            seeds.append(q)
            if labels[p] == -1:
                labels[p] = cluster_id
            elif labels[p] < 0:
                labels[p] = cluster_id
        cluster_id += 1
    # Build clusters from labels
    clusters: Dict[int,List[int]] = {}
    for idx,lbl in enumerate(labels):
        if lbl >= 0:
            clusters.setdefault(lbl, []).append(idx)
    return list(clusters.values())


def summarize_clusters(clusters: List[List[int]]) -> Dict[str, object]:
    sizes = sorted([len(c) for c in clusters], reverse=True)
    return {
        'num_clusters': len(clusters),
        'sizes_top10': sizes[:10],
        'largest': sizes[0] if sizes else 0,
        'mean_size': float(np.mean(sizes)) if sizes else 0.0,
        'median_size': float(np.median(sizes)) if sizes else 0.0
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tammy', default='Test_images/Tammy')
    ap.add_argument('--all', default='Test_images')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--complete-thresholds', default='0.40,0.45,0.50,0.55,0.60,0.65')
    ap.add_argument('--dbscan-eps', default='0.45,0.50,0.55,0.60')
    ap.add_argument('--dbscan-min-samples', type=int, default=2)
    args = ap.parse_args()

    detector = FaceDetector(device=args.device)
    recognizer = FaceRecognitionModel(model_name='arcface', device=args.device if args.device.startswith('cuda') else 'cpu')

    tammy_imgs = load_images(args.tammy)
    all_imgs = load_images(args.all)
    tammy_set = set(p for p,_ in tammy_imgs)
    full_imgs = [x for x in all_imgs if x[0] not in tammy_set]

    print(f'Tammy images: {len(tammy_imgs)} | other images: {len(full_imgs)}')

    paths, embs = compute_embeddings(detector, recognizer, full_imgs)
    if embs.shape[0] == 0:
        print('No embeddings computed for non-Tammy images; aborting.')
        return
    dist = pairwise_l2(embs)

    # Complete-link threshold sweep
    print('\nComplete-link clustering sweep:')
    thresholds = [float(t) for t in args.complete_thresholds.split(',') if t.strip()]
    for th in thresholds:
        clusters = complete_link_clusters(dist, th)
        summary = summarize_clusters(clusters)
        # Count clusters with 2-4 images
        small = sum(1 for c in clusters if 2 <= len(c) <= 4)
        print(f'Th={th:.2f} -> clusters={summary["num_clusters"]} small(2-4)={small} size_top10={summary["sizes_top10"]}')

    # DBSCAN-like sweep
    print('\nDBSCAN-like clustering sweep:')
    eps_vals = [float(e) for e in args.dbscan_eps.split(',') if e.strip()]
    for eps in eps_vals:
        clusters = dbscan_like(dist, eps, args.dbscan_min_samples)
        summary = summarize_clusters(clusters)
        small = sum(1 for c in clusters if 2 <= len(c) <= 4)
        print(f'eps={eps:.2f} -> clusters={summary["num_clusters"]} small(2-4)={small} size_top10={summary["sizes_top10"]}')

if __name__ == '__main__':
    main()
