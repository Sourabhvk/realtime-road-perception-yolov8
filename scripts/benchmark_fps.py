from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return [source]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 inference FPS.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model path or name.")
    parser.add_argument(
        "--source",
        default="data/road_dataset/images/val",
        help="Image file or directory to benchmark.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default=None, help="Device id, cuda string, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    sources = iter_sources(source_path)

    if not sources:
        raise FileNotFoundError(f"No benchmark images found at {source_path}")

    model = YOLO(args.model)

    for warmup_source in sources[:1]:
        model.predict(source=str(warmup_source), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)

    start_time = time.perf_counter()
    for image_path in sources:
        model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    elapsed = time.perf_counter() - start_time

    fps = len(sources) / elapsed if elapsed > 0 else float("inf")
    print(f"Processed {len(sources)} images in {elapsed:.2f}s -> {fps:.2f} FPS")


if __name__ == "__main__":
    main()