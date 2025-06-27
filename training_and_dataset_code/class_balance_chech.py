import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define paths
annotation_folder = 'A:/Academic/CSE498R/Dataset/Onlylebels'  # Path to the folder containing YOLO annotation files
output_image_path = 'A:/Academic/CSE498R/class_distribution.png'  # Path to save the plot image

# Define class names with "Background" as the first class
class_names = [
    "Background",  # Added for empty/background labels
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

# Initialize a dictionary to count class instances (including background)
class_counts = defaultdict(int)

# Count empty files as background
total_files = 0
empty_files = 0

# Iterate through all annotation files
for annotation_file in os.listdir(annotation_folder):
    annotation_path = os.path.join(annotation_folder, annotation_file)
    if not annotation_file.endswith('.txt'):
        continue  # Skip non-text files
    
    total_files += 1
    
    # Read the annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        
        if len(lines) == 0:  # Empty file (background)
            class_counts[0] += 1  # Class 0 is background
            empty_files += 1
            continue
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure valid YOLO annotation line
                class_id = int(parts[0]) + 1  # Shift IDs by 1 to account for background class
                class_counts[class_id] += 1

# Print some statistics
print(f"Total annotation files processed: {total_files}")
print(f"Empty annotation files (background): {empty_files}")

# Find the class with the maximum instances (excluding background if needed)
max_count = max(class_counts.values())

# Check for unbalanced classes (difference from max_count is more than 10% of max_count)
unbalanced_classes = []
for class_id, count in class_counts.items():
    difference = max_count - count
    if difference > 0.1 * max_count:  # Difference is bigger than 10% of max_count
        unbalanced_classes.append(class_id)

# Print results
print("\nClass Counts:")
for class_id in sorted(class_counts.keys()):
    count = class_counts[class_id]
    status = "Unbalanced" if class_id in unbalanced_classes else "Balanced"
    print(f"Class {class_id} ({class_names[class_id]}): {count} instances - {status}")

print("\nUnbalanced Classes (difference from max_count > 10% of max_count):")
if unbalanced_classes:
    for class_id in unbalanced_classes:
        print(f"Class {class_id} ({class_names[class_id]})")
else:
    print("All classes are balanced.")

# Prepare data for plotting
plot_class_ids = sorted(class_counts.keys())
plot_class_names = [class_names[i] for i in plot_class_ids]
plot_counts = [class_counts[i] for i in plot_class_ids]

# Generate a list of colors for each bar
colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))  # Use a colormap to generate distinct colors

# Plotting the bar chart with improved formatting
plt.figure(figsize=(14, 8))
bars = plt.bar(plot_class_names, plot_counts, color=[colors[i] for i in plot_class_ids])

plt.xlabel('Class Names', fontsize=12, fontweight='bold')
plt.ylabel('Count of Instances', fontsize=12, fontweight='bold')
plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')

# Rotate x-axis labels and adjust font size
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to make room for the rotated x-axis labels
plt.tight_layout()

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=9)

# Save the plot as a high-quality PNG image
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_image_path}")

plt.show()