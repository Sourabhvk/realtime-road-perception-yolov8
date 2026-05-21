# Purpose: dataset sanity check before training.

from pathlib import Path
import cv2

# Folders for the images, YOLO label files, and output visualizations.
image_dir = Path("data/road_dataset/images/train")
label_dir = Path("data/road_dataset/labels/train")
output_dir = Path("outputs/images")

# Map class IDs from the label files to readable object names.
class_names = {"0": "car", "1": "pedestrian", "2": "cyclist"}

# Create the output folder if it does not already exist.
output_dir.mkdir(parents=True, exist_ok=True)

# Get the first 5 image files from the training image folder.
image_paths = sorted(
    p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)[:5]

for image_path in image_paths:
    # Read the image with OpenCV.
    image = cv2.imread(str(image_path))

    # Skip the file if the image could not be loaded.
    if image is None:
        continue

    # Get the image size so normalized YOLO coordinates can be converted to pixels.
    height, width = image.shape[:2]

    # Build the matching label file path for this image.
    label_path = label_dir / f"{image_path.stem}.txt"

    if label_path.exists():
        # Read each object annotation from the label file.
        with open(label_path, "r", encoding="utf-8") as label_file:
            for line in label_file:
                # YOLO format: class_id x_center y_center box_width box_height
                parts = line.strip().split()

                # Skip invalid label lines.
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, box_width, box_height = parts

                # Convert normalized YOLO values into pixel values.
                x_center = float(x_center) * width
                y_center = float(y_center) * height
                box_width = float(box_width) * width
                box_height = float(box_height) * height

                # Convert center-based YOLO box format to corner coordinates.
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Get the readable class name, or use the class ID if it is unknown.
                name = class_names.get(class_id, class_id)

                # Draw the bounding box around the object.
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the class name above the bounding box.
                cv2.putText(
                    image,
                    name,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    # Save the visualized image with the same filename in the output folder.
    cv2.imwrite(str(output_dir / image_path.name), image)

print(f"Saved visualized images to: {output_dir}")
