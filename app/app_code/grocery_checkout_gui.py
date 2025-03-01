import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QScrollArea, QFrame, QSlider, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QFontDatabase, QIcon
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtMultimedia import QSound
from ultralytics import YOLO
from app.app_code.product_card import ProductCard  # Import the ProductCard class
from app.app_code.custom_button import CustomButton  # Import the reusable button

# Load YOLO Model
model = YOLO("feb13_v11_reg_best.pt")  # Change to your trained YOLOv11 model

# Define colors for different classes
CLASS_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#FFA500", "#800080", "#00FFFF", "#FF00FF",
    "#808000", "#008080", "#800000", "#008000",
    "#000080", "#C0C0C0", "#404040", "#FF1493",
    "#E38800FF", "#CB4242FF", "#8327CAFF",
]

# Product details (Replace this with a real database or CSV file)
PRODUCT_DETAILS = {
    0: {"name": "Chocolate Digestive Biscuit", "weight": "137g", "info": "Food and Bevarage", "barcode": "8941194003717"},
    1: {"name": "BelleAme Digestive Biscuit", "weight": "135g", "info": "Food and Bevarage", "barcode": "8941160034059"},
    2: {"name": "Mango Juice with Basil Seed", "weight": "290ml", "info": "Food and Bevarage", "barcode": "8936029050103"},
    3: {"name": "7up Drinks bottle", "weight": "500ml", "info": "Food and Bevarage", "barcode": "8941100313435"},
    4: {"name": "Good Knight Liquid Mosquito repellent refill", "weight": "45ml", "info": "Food and Bevarage", "barcode": "745110769262"},
    5: {"name": "Fresh Toilet Tissue White", "weight": "80g", "info": "Personal care and Hygene", "barcode": "8941161004914"},
    6: {"name": "Clemon Can", "weight": "250ml", "info": "Food and Bevarage", "barcode": "8941189600099"},
    7: {"name": "Mojo Can", "weight": "250ml", "info": "Food and Bevarage", "barcode": "8941189600020"},
    8: {"name": "Speed bottle", "weight": "250ml", "info": "Food and Bevarage", "barcode": "4941189600266"},
    9: {"name": "Nescafe Classsic glass jar small", "weight": "24g", "info": "Food and Bevarage", "barcode": "8901058001617"},
    10: {"name": "Ahmed canned corn-flour", "weight": "150g", "info": "Food and Bevarage", "barcode": "8823122503608"},
    11: {"name": "Genial Nature Masala Tea", "weight": "80g 40pcs", "info": "Food and Bevarage", "barcode": "8941158281199"},
    12: {"name": "Jumbo Vegetable Spring Roll", "weight": "400g 10pcs", "info": "Food and Bevarage", "barcode": "2503201600928"},
    13: {"name": "Orange Juice with Basil Seed", "weight": "290ml", "info": "Food and Bevarage", "barcode": "8935330207190"},
    14: {"name": "Ghee Premium", "weight": "200g", "info": "Food and Bevarage", "barcode": "831730007423"},
    15: {"name": "Dettol Skincare Refil Liquid Handwash", "weight": "170ml", "info": "Peronal care and Hygene", "barcode": "8941102833375"},
    16: {"name": "Whitening Mouthwash WhitePlus", "weight": "250ml", "info": "Peronal care and hygene", "barcode": "8941100503096"},
    17: {"name": "club de nuit man perfume body spray", "weight": "200ml", "info": "Personal Care and Hygene", "barcode": "6085010094335"},
    18: {"name": "Mr. Noodles Easy Instant Noodles magic masala", "weight": "400g 8pcs", "info": "Food and Bevarage", "barcode": "840205750849"},
}

class YOLOThread(QThread):
    result_signal = pyqtSignal(np.ndarray, list)  # Emit annotated image + detected products

    def __init__(self, frame):
        super().__init__()
        self.frame = frame

    def run(self):
        # Resize the frame to 960x720
        self.frame = cv2.resize(self.frame, (960, 720))

        # Perform inference
        results = model(self.frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box (xmin, ymin, xmax, ymax)
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class labels

        detected_products = []

        # Draw Bounding Boxes on image
        annotated_frame = self.draw_bboxes(self.frame.copy(), boxes, class_ids, confidences, detected_products)
        self.result_signal.emit(annotated_frame, detected_products)  # Send processed frame & detected products list

    def draw_bboxes(self, frame, boxes, class_ids, confidences, detected_products):
        """Draws bounding boxes and saves detected products."""
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x_min, y_min, x_max, y_max = map(int, box)
            color = QColor(CLASS_COLORS[class_id % len(CLASS_COLORS)])  # Assign color per class

            # Product Details
            product_name = PRODUCT_DETAILS.get(class_id, {}).get("name", f"Product {class_id}")
            weight = PRODUCT_DETAILS.get(class_id, {}).get("weight", "Unknown")
            info = PRODUCT_DETAILS.get(class_id, {}).get("info", "No info available")
            barcode = PRODUCT_DETAILS.get(class_id, {}).get("barcode", "N/A")

            detected_products.append({
                "class_id": class_id,
                "name": product_name,
                "weight": weight,
                "info": info,
                "barcode": barcode,
                "box": (x_min, y_min, x_max, y_max)  # Store box for removal
            })

            # Draw bounding box with thicker lines
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color.getRgb()[:3], thickness=3)  # Increase thickness to 3

            # Add Label with fill and dynamic text color
            label = f"{product_name} ({weight})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x_min + (x_max - x_min - text_size[0]) // 2
            text_y = y_min + (y_max - y_min + text_size[1]) // 2

            # Ensure label is visible with white background
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text

        return frame

class CameraSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Main layout
        layout = QVBoxLayout()

        # Hue Slider
        self.hue_slider = self.create_slider("Hue", 0, 179, 0)  # Hue range: 0-179
        layout.addWidget(self.hue_slider)

        # Saturation Slider
        self.saturation_slider = self.create_slider("Saturation", 0, 255, 255)  # Saturation range: 0-255
        layout.addWidget(self.saturation_slider)

        # Brightness Slider
        self.brightness_slider = self.create_slider("Brightness", -100, 100, 0)  # Brightness range: -100 to 100
        layout.addWidget(self.brightness_slider)

        # Contrast Slider
        self.contrast_slider = self.create_slider("Contrast", -100, 100, 0)  # Contrast range: -100 to 100
        layout.addWidget(self.contrast_slider)

        # Invert Colors and Black & White Toggles
        toggle_layout = QHBoxLayout()

        self.invert_colors_toggle = QCheckBox("Invert Colors")
        self.invert_colors_toggle.setStyleSheet("color: white;")
        toggle_layout.addWidget(self.invert_colors_toggle)

        self.black_white_toggle = QCheckBox("Black & White")
        self.black_white_toggle.setStyleSheet("color: white;")
        toggle_layout.addWidget(self.black_white_toggle)

        layout.addLayout(toggle_layout)

        self.setLayout(layout)

    def create_slider(self, label_text, min_value, max_value, default_value):
        """Creates a slider with a label."""
        slider_layout = QVBoxLayout()

        # Label
        label = QLabel(label_text)
        label.setStyleSheet("color: white;")
        slider_layout.addWidget(label)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #404040;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
        """)
        slider_layout.addWidget(slider)

        # Container widget
        container = QWidget()
        container.setLayout(slider_layout)
        return container

    def get_hue(self):
        """Returns the current hue value."""
        return self.hue_slider.findChild(QSlider).value()

    def get_saturation(self):
        """Returns the current saturation value."""
        return self.saturation_slider.findChild(QSlider).value()

    def get_brightness(self):
        """Returns the current brightness value."""
        return self.brightness_slider.findChild(QSlider).value()

    def get_contrast(self):
        """Returns the current contrast value."""
        return self.contrast_slider.findChild(QSlider).value()

    def is_inverted(self):
        """Returns whether the colors are inverted."""
        return self.invert_colors_toggle.isChecked()

    def is_black_white(self):
        """Returns whether the feed is in black and white."""
        return self.black_white_toggle.isChecked()

class GroceryCheckoutApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)  # Open camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Refresh every 30ms
        self.detected_products = {}  # Dictionary to store detected products and their counts
        self.current_frame = None  # Store the current frame
        self.last_annotated_frame = None  # Store the last annotated frame

    def initUI(self):
        self.setWindowTitle("AI Grocery Checkout System")
        self.setGeometry(0, 0, 1920, 1080)

        # Set background color of the main window to dark grey
        self.setStyleSheet("background-color: #2C3035;")

        # Load custom font
        font_id = QFontDatabase.addApplicationFont("AtkinsonHyperlegibleMono-Regular.ttf")  # Replace with the path to your font file
        if font_id == -1:
            print("Failed to load custom font.")
        else:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            self.custom_font = QFont(font_family, 14)  # Set font size

        # Set app icon
        self.setWindowIcon(QIcon("app_icon.png"))  # Replace with the path to your app icon

        # Main Layout
        main_layout = QGridLayout()

        # Company Logo (Top-left corner)
        logo_label = QLabel(self)
        logo_pixmap = QPixmap("company_logo.png")  # Replace with the path to your company logo
        logo_label.setPixmap(logo_pixmap.scaled(200, 50, Qt.KeepAspectRatio))  # Adjust size as needed
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        main_layout.addWidget(logo_label, 0, 0, 1, 1)  # Top-left corner

        # Live Camera View
        live_camera_label = QLabel("Live Camera View")
        live_camera_label.setFont(QFont(font_family, 10))  # Smaller font size
        live_camera_label.setStyleSheet("color: white;")
        live_camera_label.setAlignment(Qt.AlignLeft)  # Left-aligned
        main_layout.addWidget(live_camera_label, 1, 0)

        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(320, 240)  # Smaller size for live camera view
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 5px solid #638D6F;
                border-radius: 15px;
                background-color: black;
            }
        """)
        main_layout.addWidget(self.camera_label, 2, 0)

        # Camera Settings Section
        self.camera_settings = CameraSettings()
        main_layout.addWidget(self.camera_settings, 3, 0)  # Add camera settings under the live camera view

        # Reset Button
        self.reset_button = CustomButton(" Reset", color="#FF0000", hover_color="#FF5555", icon="reset_icon.png")
        self.reset_button.setFont(self.custom_font)  # Apply custom font
        self.reset_button.clicked.connect(self.reset_ui)
        main_layout.addWidget(self.reset_button, 4, 0)  # Add reset button below camera settings

        # Detected Image View
        detected_image_label = QLabel("Detected Image")
        detected_image_label.setFont(QFont(font_family, 10))  # Smaller font size
        detected_image_label.setStyleSheet("color: white;")
        detected_image_label.setAlignment(Qt.AlignLeft)  # Left-aligned
        main_layout.addWidget(detected_image_label, 1, 1)

        self.scanned_label = QLabel(self)
        self.scanned_label.setFixedSize(960, 720)  # Larger size for detected image view
        self.scanned_label.setStyleSheet("""
            QLabel {
                border: 5px solid orange;
                border-radius: 15px;
                background-color: black;
            }
        """)
        main_layout.addWidget(self.scanned_label, 2, 1)

        # Scan and Save Buttons (Moved under the detected image section and aligned horizontally with the reset button)
        button_layout = QHBoxLayout()

        # Scan Button
        self.scan_button = CustomButton(" Scan", icon="scan_icon.png")
        self.scan_button.setFont(self.custom_font)  # Apply custom font
        self.scan_button.clicked.connect(self.scan_image)
        button_layout.addWidget(self.scan_button)

        # Save Button
        self.save_button = CustomButton(" Save", color="#007608", hover_color="#029C21", icon="input_icon.png")
        self.save_button.setFont(self.custom_font)  # Apply custom font
        self.save_button.clicked.connect(self.save_results)
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(button_layout, 4, 1)  # Add buttons under the detected image section

        # Detected Products Section (Scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #23272e;")  # Set scroll area background to dark grey
        scroll_area.setFixedWidth(525)  # Change width as needed

        self.product_container = QWidget()
        self.product_layout = QVBoxLayout(self.product_container)  # Single column layout
        scroll_area.setWidget(self.product_container)

        main_layout.addWidget(scroll_area, 1, 2, 3, 1)  # Span across three rows

        self.setLayout(main_layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply camera settings
            frame = self.apply_camera_settings(frame)

            # Convert to black and white if enabled
            if self.camera_settings.is_black_white():
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # Scale down the frame for the live camera view
            small_frame = cv2.resize(frame, (320, 240))
            small_image = QImage(small_frame.data, small_frame.shape[1], small_frame.shape[0], QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(small_image))

            # Store the current frame for scanning
            self.current_frame = frame

    def apply_camera_settings(self, frame):
        """Applies camera settings (hue, saturation, brightness, contrast, and inversion) to the frame."""
        # Convert to HSV for hue and saturation adjustments
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Adjust hue and saturation
        hsv_frame[:, :, 0] = (hsv_frame[:, :, 0] + self.camera_settings.get_hue()) % 180  # Hue is cyclic (0-179)
        hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1] + self.camera_settings.get_saturation() - 255, 0, 255)

        # Convert back to RGB
        frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)

        # Adjust brightness and contrast
        alpha = (self.camera_settings.get_contrast() + 100) / 100  # Contrast scaling factor
        beta = self.camera_settings.get_brightness()  # Brightness offset
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Invert colors if enabled
        if self.camera_settings.is_inverted():
            frame = cv2.bitwise_not(frame)

        return frame

    def scan_image(self):
        if hasattr(self, 'current_frame'):
            self.yolo_thread = YOLOThread(self.current_frame.copy())
            self.yolo_thread.result_signal.connect(self.display_result)
            self.yolo_thread.start()

    def display_result(self, annotated_frame, detected_products):
        # Store the annotated frame for later use
        self.last_annotated_frame = annotated_frame

        # Convert annotated frame to QImage
        annotated_image = QImage(annotated_frame.data, annotated_frame.shape[1], annotated_frame.shape[0], QImage.Format_RGB888)
        
        # Check if the annotated_image is valid
        if annotated_image.isNull():
            print("Error: Annotated image is null.")
            return

        # Display the annotated image in the scanned_label
        self.scanned_label.setPixmap(QPixmap.fromImage(annotated_image))

        # Update the detected products dictionary
        for product in detected_products:
            class_id = product["class_id"]
            if class_id in self.detected_products:
                # If the product already exists, update its count
                self.detected_products[class_id]["count"] += 1
            else:
                # If the product is new, add it to the dictionary
                self.detected_products[class_id] = product
                self.detected_products[class_id]["count"] = 1

        # Clear the product layout
        self.clear_product_layout()

        # Add detected products to the product layout
        for class_id, product in self.detected_products.items():
            # Create a ProductCard instance with the count
            card = ProductCard(product, CLASS_COLORS, product["count"])
            
            # Connect the remove_signal to the remove_product slot
            card.remove_signal.connect(self.remove_product)
            
            # Connect the count_changed_signal to the update_product_count slot
            card.count_changed_signal.connect(self.update_product_count)
            
            # Add card to the product layout
            self.product_layout.addWidget(card)

        # Play sound to indicate scanning is complete
        QSound.play("scan_complete.wav")

    def remove_product(self, class_id):
        """Removes the product with the given class_id from the detected_products dictionary."""
        if class_id in self.detected_products:
            del self.detected_products[class_id]
        
        # Re-display the results to update the UI
        self.clear_product_layout()
        for class_id, product in self.detected_products.items():
            card = ProductCard(product, CLASS_COLORS, product["count"])
            card.remove_signal.connect(self.remove_product)
            card.count_changed_signal.connect(self.update_product_count)
            self.product_layout.addWidget(card)

    def clear_product_layout(self):
        """Clears all widgets from the product layout."""
        for i in reversed(range(self.product_layout.count())):
            self.product_layout.itemAt(i).widget().setParent(None)

    def save_results(self):
        """Saves the detected products to a file and resets the UI."""
        with open("detected_products.txt", "w") as f:
            for product in self.detected_products.values():
                barcode = product.get("barcode", "N/A")
                count = product.get("count", 1)  # Get the count, default to 1 if not present
                for _ in range(count):  # Write the barcode as many times as the count
                    f.write(f"{barcode}\n")
        print("✅ Results saved!")

        # Play sound to indicate results are saved
        QSound.play("beep.wav")

        # Reset the UI
        self.reset_ui()

    def reset_ui(self):
        """Resets the UI to a clean state."""
        # Clear the detected products dictionary
        self.detected_products.clear()

        # Clear the product layout
        self.clear_product_layout()

        # Clear the scanned image
        self.scanned_label.clear()

        # Clear the last annotated frame
        self.last_annotated_frame = None

    def update_product_count(self, class_id, new_count):
        """Updates the count of a product in the detected_products dictionary."""
        if class_id in self.detected_products:
            self.detected_products[class_id]["count"] = new_count

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GroceryCheckoutApp()
    window.show()
    sys.exit(app.exec_())