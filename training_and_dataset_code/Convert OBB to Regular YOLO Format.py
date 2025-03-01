import os
import numpy as np

def convert_obb_to_xywh(obb_file, output_folder):
    """
    Convert OBB annotation format to YOLO (class x_center y_center width height)
    """
    with open(obb_file, "r") as f:
        lines = f.readlines()

    new_annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = parts[0]
        points = np.array(parts[1:], dtype=float).reshape(4, 2)  # (x1, y1, x2, y2, x3, y3, x4, y4)

        # Get the bounding box coordinates (xmin, ymin, xmax, ymax)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Convert to YOLO format (x_center, y_center, width, height)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        new_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Save the new annotations
    output_file = os.path.join(output_folder, os.path.basename(obb_file))
    with open(output_file, "w") as f:
        f.writelines(new_annotations)

    print(f"✅ Converted: {obb_file} → {output_file}")

# Example usage
obb_annotations_folder = "A:/Academic/CSE498R/Dataset/Label/obb_labels"
yolo_annotations_folder = "A:/Academic/CSE498R/Dataset/Label/Labels"

os.makedirs(yolo_annotations_folder, exist_ok=True)
for obb_file in os.listdir(obb_annotations_folder):
    if obb_file.endswith(".txt"):
        convert_obb_to_xywh(os.path.join(obb_annotations_folder, obb_file), yolo_annotations_folder)

print("✅ Conversion Complete!")
