"""Command Line Interface for luna_face_recog.

Provides convenient commands:
  luna-face verify <img1> <img2>
  luna-face represent <img>
  luna-face detect <img>
  luna-face find <query_img> <folder>

Optional flags:
  --model arcface|facenet|vggface (default arcface)
  --device cpu|cuda|auto (default auto)
  --limit N  (limit output rows for find)

Example:
  luna-face verify tests/dataset/img1.jpg tests/dataset/img1.jpg
  luna-face represent tests/dataset/img1.jpg
  luna-face detect tests/dataset/img1.jpg
  luna-face find tests/dataset/img1.jpg tests/dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .luna_face import LunaFace
from .detection.face_detector import FaceDetector  # re-export pack info indirectly


def _print_json(data: Any):  # pretty print helper
    print(json.dumps(data, indent=2, default=float))


def cmd_verify(args):
    lf = LunaFace(model_name=args.model, device=args.device, detector_pack=args.pack, detector_root=args.pack_root)
    res = lf.verify(args.img1, args.img2)
    _print_json(res)


def cmd_represent(args):
    lf = LunaFace(model_name=args.model, device=args.device, detector_pack=args.pack, detector_root=args.pack_root)
    emb = lf.represent(args.img)
    print("Embedding length:", len(emb))
    if args.print:
        print(json.dumps(emb.tolist()))


def cmd_detect(args):
    lf = LunaFace(model_name=args.model, device=args.device, detector_pack=args.pack, detector_root=args.pack_root)
    detections = lf.detect(args.img)
    _print_json(detections)


def cmd_find(args):
    lf = LunaFace(model_name=args.model, device=args.device, detector_pack=args.pack, detector_root=args.pack_root)
    matches = lf.find(args.query, args.folder)
    limit = args.limit or len(matches)
    _print_json(matches[:limit])


def build_parser():
    p = argparse.ArgumentParser(prog="luna-face", description="Luna Face Recognition CLI")
    p.add_argument("--model", default="arcface", help="Embedding model name (arcface|facenet|vggface)")
    p.add_argument("--device", default="auto", help="Device (cpu|cuda|auto)")
    # Note: FaceDetector now auto-uses selected pack based on environment; we expose a hint flag for future use.
    p.add_argument("--pack", default="buffalo_l", help="InsightFace model pack for detection (buffalo_l|antelopev2)")
    p.add_argument("--pack-root", default=None, help="Optional root dir containing model packs (defaults to insightface cache)")
    sub = p.add_subparsers(dest="command", required=True)

    pv = sub.add_parser("verify", help="Verify two images are same person")
    pv.add_argument("img1")
    pv.add_argument("img2")
    pv.set_defaults(func=cmd_verify)

    pr = sub.add_parser("represent", help="Get embedding for an image")
    pr.add_argument("img")
    pr.add_argument("--print", action="store_true", help="Print raw embedding JSON array")
    pr.set_defaults(func=cmd_represent)

    pd = sub.add_parser("detect", help="Run face detection")
    pd.add_argument("img")
    pd.set_defaults(func=cmd_detect)

    pf = sub.add_parser("find", help="Search a folder for similar faces")
    pf.add_argument("query")
    pf.add_argument("folder")
    pf.add_argument("--limit", type=int, default=10)
    pf.set_defaults(func=cmd_find)

    return p


def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
