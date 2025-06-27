from ultralytics import YOLO
import time
import numpy as np

# Load the model
model = YOLO("yolov8n.pt")  # Replace with your model path

# Test image (replace with your image path)
image_path = "test.jpg"

# Warm-up runs
for _ in range(5):
    _ = model(image_path)

# Measure inference time
num_runs = 100
inference_times = []

for _ in range(num_runs):
    start_time = time.time()
    results = model(image_path)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    inference_times.append(inference_time)

# Remove outliers (first few runs)
inference_times = inference_times[5:]

# Calculate average and std deviation
avg_time = np.mean(inference_times)
std_time = np.std(inference_times)

print(f"Average Inference Time: {avg_time:.2f} ms ± {std_time:.2f}")