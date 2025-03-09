import json
import numpy as np

def convert_rotated_bbox_to_axis_aligned(bbox, rotation):
    """
    Convert a rotated bounding box to an axis-aligned bounding box.
    
    Parameters:
    - bbox: List [x, y, width, height] (COCO format)
    - rotation: Rotation angle in degrees
    
    Returns:
    - List [xmin, ymin, xmax, ymax] (axis-aligned format)
    """
    x, y, w, h = bbox
    rotation_rad = np.radians(rotation)

    # Compute the four corners of the rotated rectangle
    corners = np.array([
        [x, y], 
        [x + w, y], 
        [x, y + h], 
        [x + w, y + h]
    ])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad), np.cos(rotation_rad)]
    ])

    # Rotate corners around the center
    center = np.array([x + w / 2, y + h / 2])
    rotated_corners = np.dot(corners - center, rotation_matrix.T) + center

    # Get new bounding box
    xmin, ymin = rotated_corners.min(axis=0)
    xmax, ymax = rotated_corners.max(axis=0)

    return [xmin, ymin, xmax, ymax]

# Example usage
annotation_path = "C:/Users/Akash/Downloads/dataset in coco format/annotations/instances_default.json"
with open(annotation_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

for annotation in coco_data["annotations"]:
    if "rotation" in annotation["attributes"]:
        rotation = annotation["attributes"]["rotation"]
        bbox = annotation["bbox"]
        annotation["bbox"] = convert_rotated_bbox_to_axis_aligned(bbox, rotation)

# Save updated annotations
with open("C:/Users/Akash/Downloads/dataset in coco format/annotations/converted_annotations.json", "w", encoding="utf-8") as f:
    json.dump(coco_data, f, indent=4)

print("Converted annotations saved as 'converted_annotations.json'.")
