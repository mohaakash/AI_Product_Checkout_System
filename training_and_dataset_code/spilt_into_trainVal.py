import os
import random
import shutil

# Define paths
image_folder = 'path/to/images'  # Path to the folder containing images
label_folder = 'path/to/labels'  # Path to the folder containing YOLO annotation files
output_folder = 'path/to/output'  # Path to the folder where train and val folders will be created

# Create output directories
train_image_folder = os.path.join(output_folder, 'train', 'images')
train_label_folder = os.path.join(output_folder, 'train', 'labels')
val_image_folder = os.path.join(output_folder, 'val', 'images')
val_label_folder = os.path.join(output_folder, 'val', 'labels')

os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

# Get list of image files (assuming images are in .jpg format)
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
random.shuffle(image_files)  # Shuffle the list to ensure random splitting

# Split ratio (80% train, 20% validation)
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# Split into train and validation sets
train_images = image_files[:split_index]
val_images = image_files[split_index:]

# Function to copy files to their respective folders
def copy_files(file_list, src_image_folder, src_label_folder, dst_image_folder, dst_label_folder):
    for file in file_list:
        # Copy image
        image_src = os.path.join(src_image_folder, file)
        image_dst = os.path.join(dst_image_folder, file)
        shutil.copy(image_src, image_dst)
        
        # Copy corresponding label
        label_file = file.replace('.jpg', '.txt')  # Assuming labels have the same name as images but with .txt extension
        label_src = os.path.join(src_label_folder, label_file)
        label_dst = os.path.join(dst_label_folder, label_file)
        if os.path.exists(label_src):  # Ensure the label file exists
            shutil.copy(label_src, label_dst)

# Copy train files
copy_files(train_images, image_folder, label_folder, train_image_folder, train_label_folder)

# Copy validation files
copy_files(val_images, image_folder, label_folder, val_image_folder, val_label_folder)

print(f"Dataset split completed:")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")