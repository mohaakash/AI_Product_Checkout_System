import os
import random
import shutil
import cv2

# Paths
dataset_dir = "C:/Users/Akash/Downloads/Telegram Desktop/afterSplit/afterSplit/train/images"  # Folder where original images are stored
labels_dir = "C:/Users/Akash/Downloads/Telegram Desktop/afterSplit/afterSplit/train/labels"   # Folder where YOLO labels are stored
output_dir = "C:/Users/Akash/Downloads/Telegram Desktop/afterSplit/afterSplit/train/images"  # Output folder for grayscale images
output_labels_dir = "C:/Users/Akash/Downloads/Telegram Desktop/afterSplit/afterSplit/train/labels"  # Output folder for copied labels

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Select 10% randomly
num_to_convert = int(len(image_files) * 0.3)
selected_images = random.sample(image_files, num_to_convert)

# Process each selected image
for img_name in selected_images:
    img_path = os.path.join(dataset_dir, img_name)
    label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))
    
    # Load image and convert to grayscale
    image = cv2.imread(img_path)
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert single-channel grayscale to 3-channel to match model input expectations
    grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)

    # Save grayscale image with a modified name (e.g., "image_001_grey.jpg")
    new_img_name = os.path.splitext(img_name)[0] + "_grey.jpg"
    cv2.imwrite(os.path.join(output_dir, new_img_name), grayscale_img)

    # Copy the corresponding label file
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_labels_dir, new_img_name.replace(".jpg", ".txt")))

print(f"Converted {num_to_convert} images to grayscale and saved them in {output_dir}")
