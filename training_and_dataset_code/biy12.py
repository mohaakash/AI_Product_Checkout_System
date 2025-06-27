import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# ===== CONFIGURATION =====
INPUT_FOLDER = "A:/Academic/CSE498R/Dataset/test/photo"  # Folder containing images to process
OUTPUT_FOLDER = "A:/Academic/CSE498R/Dataset/test/result"  # Folder to save processed images
CONVERT_TO_GRAYSCALE = True           # Convert images to grayscale before inference
model = YOLO("app/models/12m.pt")  # Change to your trained YOLO model
CONFIDENCE_THRESHOLD = 0.5            # Minimum confidence score to keep detection
MODEL_PATH = "app/models/12m.pt"  # Path to the YOLO model

# Define colors for different classes (19 distinct colors)
CLASS_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500",
    "#800080", "#00FFFF", "#FF00FF", "#808000", "#008080",
    "#800000", "#008000", "#000080", "#C0C0C0", "#404040",
    "#FF1493", "#E38800", "#CB4242", "#8327CA"
]

# ===== HELPER FUNCTIONS =====
def load_product_details(csv_file="product_details.csv"):
    """Load product details from CSV file if available"""
    product_details = {}
    if os.path.exists(csv_file):
        import csv
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
        print(f"Loaded product details for {len(product_details)} classes")
    else:
        print("No product_details.csv found - using default class IDs")
    return product_details

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Convert class colors to RGB
CLASS_COLORS = [hex_to_rgb(color) for color in CLASS_COLORS]

# Load product details
PRODUCT_DETAILS = load_product_details()

# ===== DETECTION FUNCTIONS =====
def extract_detections(results):
    """
    Universal detection extractor compatible with YOLOv8 and future versions
    Returns: (boxes, confidences, class_ids)
    """
    # YOLOv8 style
    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    # Hypothetical YOLOv12 style (if API changes)
    # elif hasattr(results, 'detections'):
    #     boxes = results.detections.xyxy
    #     confidences = results.detections.confidence
    #     class_ids = results.detections.class_id
    else:
        raise ValueError("Unsupported results format - check YOLO version")
    
    return boxes, confidences, class_ids

def draw_detections(image, boxes, class_ids, confidences):
    """Draw bounding boxes and labels on image"""
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        if conf < CONFIDENCE_THRESHOLD:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Get product info
        product_name = PRODUCT_DETAILS.get(class_id, {}).get("name", f"Class {class_id}")
        weight = PRODUCT_DETAILS.get(class_id, {}).get("weight", "")
        
        # Create label
        label = f"{product_name} {weight}".strip()
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

# ===== IMAGE PROCESSING =====
def preprocess_image(image):
    """Convert and resize image for processing"""
    if CONVERT_TO_GRAYSCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize while maintaining aspect ratio
    height, width = image.shape[:2]
    new_height = 720
    new_width = int(width * (new_height / height))
    return cv2.resize(image, (new_width, new_height))

# ===== MAIN PROCESSING =====
def process_image(image_path, model):
    """Process a single image through the model"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None, 0
    
    processed_image = preprocess_image(image)
    
    # Run inference
    start_time = time.time()
    results = model(processed_image)
    inference_time = time.time() - start_time
    
    # Extract detections
    boxes, confidences, class_ids = extract_detections(results)
    
    # Draw detections on original image (not resized)
    annotated_image = draw_detections(image, boxes, class_ids, confidences)
    
    return annotated_image, inference_time

def process_all_images():
    """Batch process all images in input folder"""
    # Create folders if needed
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Get image files
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    total_time = 0
    processed_count = 0
    
    for image_file in image_files:
        input_path = os.path.join(INPUT_FOLDER, image_file)
        output_path = os.path.join(OUTPUT_FOLDER, image_file)
        
        print(f"Processing {image_file}...", end=" ", flush=True)
        
        try:
            result, inference_time = process_image(input_path, model)
            if result is not None:
                cv2.imwrite(output_path, result)
                total_time += inference_time
                processed_count += 1
                print(f"done in {inference_time:.2f}s")
            else:
                print("failed (no result)")
        except Exception as e:
            print(f"failed with error: {str(e)}")
    
    # Print summary
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"\nProcessing complete! {processed_count}/{len(image_files)} images processed successfully")
        print(f"Average processing time: {avg_time:.2f} seconds per image")
        print(f"Results saved to: {os.path.abspath(OUTPUT_FOLDER)}")
    else:
        print("\nNo images were successfully processed")

if __name__ == "__main__":
    process_all_images()