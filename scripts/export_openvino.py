"""Export the YOLO weights to an OpenVINO IR for fast CPU/edge inference.

Produces `<stem>_openvino_model/` next to the weights (default FP16, ~4-5x
faster than PyTorch on Intel CPU with ~3% detection jitter near the confidence
threshold). Run once, or after swapping the .pt weights.

Usage:
    python scripts/export_openvino.py            # FP16 (recommended)
    python scripts/export_openvino.py --fp32     # full precision
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to OpenVINO IR")
    parser.add_argument("--fp32", action="store_true", help="export full precision (default: FP16)")
    parser.add_argument("--imgsz", type=int, default=640, help="export input size")
    args = parser.parse_args()

    weights = settings.paths.base_dir / settings.yolo.model_name
    if not weights.exists():
        print(f"Weights not found: {weights}")
        sys.exit(1)

    print(f"Exporting {weights} to OpenVINO ({'FP32' if args.fp32 else 'FP16'}, imgsz={args.imgsz})...")
    out = YOLO(str(weights)).export(format="openvino", imgsz=args.imgsz, half=not args.fp32)
    print(f"Done: {out}")
    print(f"Set yolo.backend='openvino' (the default) to use it. Expected dir: "
          f"{settings.yolo.openvino_model_dir}")


if __name__ == "__main__":
    main()
