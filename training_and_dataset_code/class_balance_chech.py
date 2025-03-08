import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define paths
annotation_folder = 'A:/Academic/CSE498R/Dataset/labels for check'  # Path to the folder containing YOLO annotation files

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

# Initialize a dictionary to count class instances
class_counts = defaultdict(int)

# Iterate through all annotation files
for annotation_file in os.listdir(annotation_folder):
    annotation_path = os.path.join(annotation_folder, annotation_file)
    if not annotation_file.endswith('.txt'):
        continue  # Skip non-text files

    # Read the annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure valid YOLO annotation line
                class_id = int(parts[0])
                class_counts[class_id] += 1

# Find the class with the maximum instances
max_count = max(class_counts.values())

# Check for unbalanced classes (difference from max_count is less than 10% of max_count)
unbalanced_classes = []
for class_id, count in class_counts.items():
    difference = max_count - count
    if difference > 0.1 * max_count:  # Difference is bigger than 10% of max_count
        unbalanced_classes.append(class_id)

# Print results
print("Class Counts:")
for class_id, count in class_counts.items():
    status = "Unbalanced" if class_id in unbalanced_classes else "Balanced"
    print(f"Class {class_id} ({class_names[class_id]}): {count} instances - {status}")

print("\nUnbalanced Classes (difference from max_count < 10% of max_count):")
if unbalanced_classes:
    for class_id in unbalanced_classes:
        print(f"Class {class_id} ({class_names[class_id]})")
else:
    print("All classes are balanced.")

# Generate a list of colors for each bar
colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))  # Use a colormap to generate distinct colors

# Plotting the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, [class_counts[i] for i in range(len(class_names))], color=colors)
plt.xlabel('Class Names')
plt.ylabel('Count of Instances')
plt.title('Class Distribution')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

# Add a legend (optional)
plt.legend(bars, class_names, bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot

plt.show()