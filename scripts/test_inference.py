from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "yolov8n.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "road_dataset" / "images" / "val"


def main() -> None:
    model = YOLO(str(DEFAULT_MODEL))
    model.predict(source=str(DEFAULT_SOURCE), conf=0.25, save=True)


if __name__ == "__main__":
    main()