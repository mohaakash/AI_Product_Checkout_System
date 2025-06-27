import cv2
import csv
import numpy as np
import os
import time
from ultralytics import YOLO

# Configuration
INPUT_FOLDER = "A:/Academic/CSE498R/Dataset/test/photo"  # Folder containing images to process
OUTPUT_FOLDER = "A:/Academic/CSE498R/Dataset/test/result"  # Folder to save processed images
CONVERT_TO_GRAYSCALE = True  # Set to True to convert images to grayscale before inference

# Load YOLO Model
model = YOLO("app/models/rtdetr.pt")  # Change to your trained YOLO model

# Define colors for different classes
CLASS_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#FFA500", "#800080", "#00FFFF", "#FF00FF",
    "#808000", "#008080", "#800000", "#008000",
    "#000080", "#C0C0C0", "#404040", "#FF1493",
    "#E38800FF", "#CB4242FF", "#8327CAFF",
]

# Load product details from CSV file
def load_product_details(csv_file):
    product_details = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_id = int(row['class_id'])
            product_details[class_id] = {
                "name": row['name'],
                "weight": row['weight'],
                "info": row['info'],
                "barcode": row['barcode']
            }
    return product_details

# Load product details from CSV (if available)
try:
    PRODUCT_DETAILS = load_product_details("product_details.csv")
except FileNotFoundError:
    PRODUCT_DETAILS = {}
    print("Warning: product_details.csv not found. Using default class IDs only.")

def process_image(image_path):
    """Process a single image and return annotated image and detection time"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, 0
    
    # Convert to grayscale if configured
    if CONVERT_TO_GRAYSCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image to 960x720
    image = cv2.resize(image, (960, 720))
    
    # Start timing
    start_time = time.time()
    
    # Perform inference
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box (xmin, ymin, xmax, ymax)
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class labels
    
    # Draw bounding boxes
    annotated_image = draw_bboxes(image.copy(), boxes, class_ids, confidences)
    
    # Calculate detection time
    detection_time = time.time() - start_time
    
    return annotated_image, detection_time

def draw_bboxes(frame, boxes, class_ids, confidences):
    """Draws bounding boxes with Roboflow-like design."""
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        x_min, y_min, x_max, y_max = map(int, box)
        color = tuple(int(CLASS_COLORS[class_id % len(CLASS_COLORS)][i:i+2], 16) for i in (1, 3, 5))  # Convert hex to RGB
        
        # Product Details
        product_name = PRODUCT_DETAILS.get(class_id, {}).get("name", f"Product {class_id}")
        weight = PRODUCT_DETAILS.get(class_id, {}).get("weight", "Unknown")
        
        # Draw semi-transparent black background for the bounding box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)  # Color fill
        alpha = .15  # Opacity (15% transparency)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw bounding box with a specific color
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness=2)
        
        # Add Label with white text on a colored background
        label = f"{product_name} ({weight})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x_min + 5  # Padding from the left
        text_y = y_min - 5  # Padding above the bounding box
        
        # Draw semi-transparent background for the text
        cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2), 
                     (text_x + text_size[0] + 2, text_y + 2), color, -1)
        alpha = 1  # Opacity (100% transparency)
        overlay = frame.copy()
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add white text
        cv2.putText(frame, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text
    
    return frame

def process_all_images():
    """Process all images in the input folder and save results to output folder"""
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    total_time = 0
    processed_count = 0
    
    for image_file in image_files:
        input_path = os.path.join(INPUT_FOLDER, image_file)
        output_path = os.path.join(OUTPUT_FOLDER, image_file)
        
        # Process image
        annotated_image, detection_time = process_image(input_path)
        
        if annotated_image is not None:
            # Save annotated image
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            total_time += detection_time
            processed_count += 1
            print(f"Processed {image_file} in {detection_time:.2f} seconds")
    
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"\nProcessing complete! {processed_count} images processed.")
        print(f"Average processing time: {avg_time:.2f} seconds per image")
    else:
        print("\nNo images were successfully processed.")

if __name__ == "__main__":
    process_all_images()