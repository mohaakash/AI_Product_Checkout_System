import os
import cv2

# Define paths
image_folder = 'A:/Academic/CSE498R/Dataset/Data'  # Path to the folder containing images
annotation_folder = 'A:/Academic/CSE498R/Dataset/Label/Labels'  # Path to the folder containing YOLO annotation files
output_folder = 'A:/Academic/CSE498R/Dataset/CheckData'  # Path to save the annotated images

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define class names
class_names = [
    "Chocolate Digestive Biscuit",
    "BelleAme Digestive Biscuit",
    "Mango Juice with Basil Seed",
    "7up Drinks bottle",
    "Good Knight Liquid Mosquito repellent refill",
    "Fresh Toilet Tissue White",
    "Clemon Can",
    "Mojo Can",
    "Speed bottle",
    "Nescafe Classic glass jar small",
    "Ahmed canned corn-flour",
    "Genial Nature Masala Tea",
    "Jumbo Vegetable Spring Roll",
    "Orange Juice with Basil Seed",
    "Ghee Premium",
    "Dettol Skincare Refill Liquid Handwash",
    "Whitening Mouthwash WhitePlus",
    "club de nuit man perfume body spray",
    "Mr. Noodles Easy Instant Noodles magic masala"
]

# Define class colors (you can add more colors if needed)
class_colors = [
    (255, 0, 0),    # Class 0: Red
    (0, 255, 0),    # Class 1: Green
    (0, 0, 255),    # Class 2: Blue
    (255, 255, 0),  # Class 3: Cyan
    (255, 0, 255),  # Class 4: Magenta
    (0, 255, 255),  # Class 5: Yellow
    (128, 0, 0),    # Class 6: Maroon
    (0, 128, 0),    # Class 7: Dark Green
    (0, 0, 128),    # Class 8: Navy
    (128, 128, 0),  # Class 9: Olive
    (128, 0, 128),  # Class 10: Purple
    (0, 128, 128),  # Class 11: Teal
    (192, 192, 192),# Class 12: Silver
    (128, 128, 128),# Class 13: Gray
    (255, 165, 0),  # Class 14: Orange
    (255, 192, 203),# Class 15: Pink
    (165, 42, 42),  # Class 16: Brown
    (0, 0, 0),      # Class 17: Black
    (255, 255, 255) # Class 18: White
]

# Set rectangle thickness
rectangle_thickness = 3  # Increase this value to make the rectangles thicker

# Iterate through all images in the image folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    annotation_path = os.path.join(annotation_folder, os.path.splitext(image_name)[0] + '.txt')

    # Check if the corresponding annotation file exists
    if not os.path.exists(annotation_path):
        print(f"Annotation file for {image_name} does not exist.")
        continue

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_name}")
        continue

    # Get image dimensions
    img_height, img_width, _ = image.shape

    # Read the annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    # Process each line in the annotation file
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # Skip invalid lines

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert YOLO format to bounding box coordinates
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # Get the color and class name for the class
        color = class_colors[class_id % len(class_colors)]
        class_name = class_names[class_id]

        # Draw the bounding box with thicker lines
        cv2.rectangle(image, (x1, y1), (x2, y2), color, rectangle_thickness)

        # Add the class name as text
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    # Save the annotated image to the output folder
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image: {output_path}")

print("Annotation verification complete.")