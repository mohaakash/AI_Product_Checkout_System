import os
import cv2
import shutil

def convert_yolo_to_faster_rcnn(src_root, dst_root):
    # Create destination directories
    os.makedirs(os.path.join(dst_root, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "val", "labels"), exist_ok=True)

    # Process train and val sets
    for split in ["train", "val"]:
        img_src_dir = os.path.join(src_root, split, "images")
        label_src_dir = os.path.join(src_root, split, "labels")
        img_dst_dir = os.path.join(dst_root, split, "images")
        label_dst_dir = os.path.join(dst_root, split, "labels")

        # Copy images
        for img_name in os.listdir(img_src_dir):
            img_src_path = os.path.join(img_src_dir, img_name)
            img_dst_path = os.path.join(img_dst_dir, img_name)
            shutil.copy(img_src_path, img_dst_path)

        # Convert and copy labels
        for label_name in os.listdir(label_src_dir):
            label_src_path = os.path.join(label_src_dir, label_name)
            label_dst_path = os.path.join(label_dst_dir, label_name)

            img_name = label_name.replace(".txt", ".jpg")  # Assuming images are JPG
            img_path = os.path.join(img_src_dir, img_name)
            
            # Check if image exists (skip labels without corresponding images)
            if not os.path.exists(img_path):
                continue

            # Read image dimensions
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            with open(label_src_path, "r") as src_file, open(label_dst_path, "w") as dst_file:
                for line in src_file.readlines():
                    values = list(map(float, line.strip().split()))
                    class_id = int(values[0])
                    x_center, y_center, w, h = values[1:]

                    # Convert to Pascal VOC (absolute pixel values)
                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height
                    x_max = (x_center + w / 2) * width
                    y_max = (y_center + h / 2) * height

                    # Ensure valid bounding box values
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(width, x_max), min(height, y_max)

                    # Save in Pascal VOC format
                    dst_file.write(f"{class_id + 1} {x_min} {y_min} {x_max} {y_max}\n")

if __name__ == "__main__":
    # Example usage
    src_root = "A:/Academic/CSE498R/Dataset/yolo_data"
    dst_root = "A:/Academic/CSE498R/Dataset/frcnn_data"
    convert_yolo_to_faster_rcnn(src_root, dst_root)
    print("Conversion completed successfully.")
