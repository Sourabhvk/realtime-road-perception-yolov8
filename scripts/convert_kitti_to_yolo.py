"""Convert the KITTI Object Detection dataset into YOLOv8 format.

This script reads KITTI label files, keeps only Car, Pedestrian, and Cyclist
objects, converts their bounding boxes to YOLO format, and writes a train/val
split under data/road_dataset/.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


CLASS_MAP = {
	"Car": 0,
	"Pedestrian": 1,
	"Cyclist": 2,
}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def find_image_path(image_dir: Path, stem: str) -> Path | None:
	"""Find the image file that matches a KITTI label stem."""

	for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
		candidate = image_dir / f"{stem}{suffix}"
		if candidate.exists():
			return candidate
	return None


def convert_bbox_to_yolo(
	left: float,
	top: float,
	right: float,
	bottom: float,
	image_width: int,
	image_height: int,
) -> tuple[float, float, float, float]:
	"""Convert KITTI box coordinates to normalized YOLO format."""

	x_center = ((left + right) / 2.0) / image_width
	y_center = ((top + bottom) / 2.0) / image_height
	width = (right - left) / image_width
	height = (bottom - top) / image_height
	return x_center, y_center, width, height


def parse_kitti_label_file(
	label_path: Path,
	image_width: int,
	image_height: int,
) -> list[str]:
	"""Parse one KITTI label file and return YOLO label lines."""

	yolo_lines: list[str] = []

	with label_path.open("r", encoding="utf-8") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			if not line:
				continue

			parts = line.split()
			if len(parts) < 8:
				continue

			class_name = parts[0]
			if class_name not in CLASS_MAP:
				continue

			try:
				left, top, right, bottom = map(float, parts[4:8])
			except ValueError:
				continue

			x_center, y_center, width, height = convert_bbox_to_yolo(
				left,
				top,
				right,
				bottom,
				image_width,
				image_height,
			)

			yolo_lines.append(
				f"{CLASS_MAP[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
			)

	return yolo_lines


def write_data_yaml(output_root: Path) -> None:
	"""Create a minimal YOLO data.yaml for the converted dataset."""

	data_yaml = output_root / "data.yaml"
	data_yaml.write_text(
		"path: data/road_dataset\n"
		"train: images/train\n"
		"val: images/val\n\n"
		"names:\n"
		"  0: car\n"
		"  1: pedestrian\n"
		"  2: cyclist\n",
		encoding="utf-8",
	)


def prepare_output_directories(output_root: Path) -> None:
	"""Create the YOLO folder structure if it does not already exist."""

	for folder in (
		output_root / "images" / "train",
		output_root / "images" / "val",
		output_root / "labels" / "train",
		output_root / "labels" / "val",
	):
		folder.mkdir(parents=True, exist_ok=True)


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	image_dir = project_root / "data" / "kitti_raw" / "data_object_image_2" / "training" / "image_2"
	label_dir = project_root / "data" / "kitti_raw" / "data_object_label_2" / "training" / "label_2"
	output_root = project_root / "data" / "road_dataset"

	if not image_dir.exists():
		raise FileNotFoundError(f"Image directory not found: {image_dir}")
	if not label_dir.exists():
		raise FileNotFoundError(f"Label directory not found: {label_dir}")

	prepare_output_directories(output_root)

	label_files = sorted(label_dir.glob("*.txt"))
	if not label_files:
		print(f"No KITTI label files found in {label_dir}")
		return

	random.Random(RANDOM_SEED).shuffle(label_files)
	train_count = int(len(label_files) * TRAIN_RATIO)
	train_files = label_files[:train_count]
	val_files = label_files[train_count:]

	total_images_processed = 0
	split_counts = {"train": 0, "val": 0}

	for split_name, split_files in (
		("train", train_files),
		("val", val_files),
	):
		image_output_dir = output_root / "images" / split_name
		label_output_dir = output_root / "labels" / split_name

		for label_path in tqdm(split_files, desc=f"Processing {split_name}"):
			image_path = find_image_path(image_dir, label_path.stem)
			if image_path is None:
				print(f"Skipping {label_path.name}: matching image not found")
				continue

			image = cv2.imread(str(image_path))
			if image is None:
				print(f"Skipping {image_path.name}: unable to read image")
				continue

			image_height, image_width = image.shape[:2]
			yolo_lines = parse_kitti_label_file(label_path, image_width, image_height)

			shutil.copy2(image_path, image_output_dir / image_path.name)
			(label_output_dir / f"{label_path.stem}.txt").write_text(
				"\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
				encoding="utf-8",
			)

			total_images_processed += 1
			split_counts[split_name] += 1

	write_data_yaml(output_root)

	print(f"Total images processed: {total_images_processed}")
	print(f"Train images: {split_counts['train']}")
	print(f"Validation images: {split_counts['val']}")


if __name__ == "__main__":
	main()
