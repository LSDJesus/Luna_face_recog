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
        best_pair = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                ci = clusters[i]
                cj = clusters[j]
                max_d = 0.0
                # compute max pairwise
                for a in ci:
                    row = dist[a]
                    for b in cj:
                        d = row[b]
                        if d > max_d:
                            max_d = d
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', default='Test_images', help='Directory with images (will exclude Tammy subdir automatically)')
    ap.add_argument('--tammy', default='Test_images/Tammy')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threshold', type=float, default=0.45, help='Complete-link L2 distance threshold')
    ap.add_argument('--pack', default='buffalo_l', help='InsightFace model pack (e.g. buffalo_l, antelopev2)')
    ap.add_argument('--pack-root', default=None, help='Optional root directory containing model packs')
    ap.add_argument('--min-cluster-size', type=int, default=2, help='Only print clusters >= this size')
    ap.add_argument('--prefix-match', action='store_true', help='Show filename prefix (before first dash) distribution per cluster')
    ap.add_argument('--metrics', action='store_true', help='Compute prefix purity / fragmentation metrics')
    args = ap.parse_args()

    detector = FaceDetector(device=args.device, model_pack=args.pack, model_root=args.pack_root)
    recognizer = FaceRecognitionModel(model_name='arcface', device=args.device if args.device.startswith('cuda') else 'cpu')

    all_imgs = load_images(args.all)
    tammy_imgs = load_images(args.tammy)
    tammy_set = set(p for p,_ in tammy_imgs)
    target_imgs = [x for x in all_imgs if x[0] not in tammy_set]
    print(f'Total images (excluding Tammy): {len(target_imgs)}')

    paths, embs = compute_embeddings(detector, recognizer, target_imgs)
    if embs.shape[0] == 0:
        print('No embeddings; abort.')
        return
    dist = pairwise_l2(embs)
    clusters = complete_link_clusters(dist, args.threshold)
    # Sort clusters by size desc
    clusters.sort(key=lambda c: len(c), reverse=True)
    print(f'Formed {len(clusters)} clusters with threshold {args.threshold}')

    def prefix_of(p: str) -> str:
        base = os.path.basename(p)
        if '-' in base:
            return base.split('-')[0]
        return base.split('.')[0]

    for idx,c in enumerate(clusters):
        if len(c) < args.min_cluster_size:
            continue
        cluster_paths = [paths[i] for i in c]
        print('='*60)
        print(f'Cluster {idx+1} | size={len(c)}')
        if args.prefix_match:
            prefs = {}
            for p in cluster_paths:
                pr = prefix_of(p)
                prefs[pr] = prefs.get(pr,0)+1
            sorted_prefs = sorted(prefs.items(), key=lambda x: (-x[1], x[0]))
            print('Prefix counts:', sorted_prefs)
        for p in cluster_paths:
            print('  ', p)

    if args.metrics:
        # Compute micro purity and fragmentation
        prefix_total: Dict[str,int] = {}
        for p in paths:
            pr = prefix_of(p)
            prefix_total[pr] = prefix_total.get(pr,0)+1
        cluster_stats = []
        micro_correct = 0
        for c in clusters:
            prefs = {}
            for i in c:
                pr = prefix_of(paths[i])
                prefs[pr] = prefs.get(pr,0)+1
            top_prefix, top_count = max(prefs.items(), key=lambda x: x[1])
            purity = top_count / len(c)
            micro_correct += top_count
            cluster_stats.append({
                'size': len(c),
                'top_prefix': top_prefix,
                'top_count': top_count,
                'purity': purity,
                'unique_prefixes': len(prefs)
            })
        micro_purity = micro_correct / len(paths)
        # Fragmentation: for each prefix, number of clusters it appears in
        prefix_clusters: Dict[str,int] = {k:0 for k in prefix_total}
        for c in clusters:
            seen = set()
            for i in c:
                seen.add(prefix_of(paths[i]))
            for pr in seen:
                prefix_clusters[pr] += 1
        avg_frag = sum(prefix_clusters.values())/len(prefix_clusters)
        multi_frag = {k:v for k,v in prefix_clusters.items() if v>1}
        print('\nMETRICS SUMMARY')
        print('---------------')
        print(f'Micro purity (sum majority counts / total images): {micro_purity:.4f}')
        macro_purity = sum(cs['purity'] for cs in cluster_stats)/len(cluster_stats)
        print(f'Macro purity (average cluster purity): {macro_purity:.4f}')
        print(f'Average clusters per prefix (fragmentation): {avg_frag:.3f}')
        if multi_frag:
            worst = sorted(multi_frag.items(), key=lambda x: -x[1])[:10]
            print('Top fragmented prefixes (prefix -> clusters):', worst)
        large_mixed = [cs for cs in cluster_stats if cs['size']>=5 and cs['purity']<0.75]
        if large_mixed:
            print(f'Large mixed clusters (<0.75 purity): {len(large_mixed)}')
        print('Done.')

if __name__ == '__main__':
    main()
