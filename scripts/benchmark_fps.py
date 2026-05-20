from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_sources(source: Path) -> list[Path]:
    # Accept either a directory of images or a single image path so the same
    # benchmarking loop can handle both use cases.
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return [source]


def parse_args() -> argparse.Namespace:
    # Keep benchmark settings configurable from the command line so you can
    # compare different models, datasets, image sizes, thresholds, and devices.
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

    # Fail early if the path is wrong or a directory contains no supported
    # image files; otherwise the FPS calculation would be misleading.
    if not sources:
        raise FileNotFoundError(f"No benchmark images found at {source_path}")

    # Loading the model is intentionally outside the timed section because this
    # benchmark measures inference speed, not startup or weight-loading time.
    model = YOLO(args.model)

    # Run one prediction before timing. This "warms up" model/device internals
    # such as CUDA kernels so the first measured image is not unusually slow.
    for warmup_source in sources[:1]:
        model.predict(source=str(warmup_source), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)

    # Time only the repeated inference calls across the selected images.
    start_time = time.perf_counter()
    for image_path in sources:
        model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    elapsed = time.perf_counter() - start_time

    # FPS is the number of processed images divided by total inference time.
    fps = len(sources) / elapsed if elapsed > 0 else float("inf")
    print(f"Processed {len(sources)} images in {elapsed:.2f}s -> {fps:.2f} FPS")


if __name__ == "__main__":
    main()
