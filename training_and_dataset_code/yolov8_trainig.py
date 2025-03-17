from ultralytics import YOLO
import torch

def train_yolo():
    # Check if CUDA (GPU) is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the YOLOv8m model
    model = YOLO('yolov8m.pt')
    
    # Train the model with mild augmentations
    model.train(
        data='data.yaml',  # Path to your dataset YAML file
        epochs=100,         # Number of training epochs
        imgsz=640,         # Input image size
        batch=8,           # Reduce batch size for GTX 1050 (adjust if needed)
        device=device,     # Use GPU if available, otherwise CPU
        workers=2,         # Reduce workers for low VRAM
        project='runs/detect',  # Save results in 'runs/detect/train/'
        name='train_yolov8m',   # Experiment name
        augment=True,      # Enable augmentation
        flipud=0.2,        # Slight vertical flip
        fliplr=0.5,        # Horizontal flip
        hsv_h=0.015,       # Slight hue shift
        hsv_s=0.5,         # Moderate saturation shift
        hsv_v=0.4,         # Brightness variation
        mosaic=0.2,        # Mild mosaic augmentation
    )

if __name__ == "__main__":
    train_yolo()
