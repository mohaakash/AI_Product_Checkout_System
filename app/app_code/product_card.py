from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsDropShadowEffect
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, pyqtSignal

class ProductCard(QFrame):
    # Define signals
    remove_signal = pyqtSignal(int)  # Emit the class_id of the product to remove
    count_changed_signal = pyqtSignal(int, int)  # Emit (class_id, new_count)

    def __init__(self, product, class_colors, count=1):
        super().__init__()
        self.product = product
        self.class_colors = class_colors
        self.count = count  # Initialize the count
        self.initUI()

    def initUI(self):
        # Border color for the card based on the product class
        border_color = self.class_colors[self.product['class_id'] % len(self.class_colors)]
        
        # Apply styling to the card
        self.setStyleSheet(f"""
            background-color: #1e2227;  /* Dark grey fill */
            border: 2px solid {border_color};  /* Only border on the card itself */
            border-radius: 12px;  /* Rounded corners */
            padding: 1px;
        """)
        self.setFixedSize(480, 130)  # Adjusted height for better spacing

        # Add shadow effect
        self.setGraphicsEffect(self.create_shadow())

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduce inner padding

        # Header layout (Product Name + Counter + Remove Button)
        header_layout = QHBoxLayout()
        
        # Product Name (Font Size 12, Bold)
        name_label = QLabel(self.product["name"])
        name_label.setFont(QFont("Arial", 12, QFont.Bold))
        name_label.setStyleSheet("color: white; border: none;")  # No border
        header_layout.addWidget(name_label)

        # Counter Layout (for +, count, -)
        counter_layout = QHBoxLayout()
        counter_layout.setSpacing(5)  # Reduce spacing between counter elements

        # Decrease Count Button (-)
        decrease_button = QPushButton("-")
        decrease_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;  /* Dark grey */
                border-radius: 12px;
                color: white;
                font-size: 12px;
                padding: 5px;
                min-width: 20px;
                min-height: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        decrease_button.setFixedSize(20, 20)
        decrease_button.clicked.connect(self.decrease_count)  # Connect to decrease handler
        counter_layout.addWidget(decrease_button)

        # Count Label
        self.count_label = QLabel(str(self.count))
        self.count_label.setFont(QFont("Arial", 12))
        self.count_label.setStyleSheet("color: white; border: none;")  # No border
        counter_layout.addWidget(self.count_label)

        # Increase Count Button (+)
        increase_button = QPushButton("+")
        increase_button.setStyleSheet("""
            QPushButton {
                background-color: #404040;  /* Dark grey */
                border-radius: 12px;
                color: white;
                font-size: 12px;
                padding: 5px;
                min-width: 20px;
                min-height: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        increase_button.setFixedSize(20, 20)
        increase_button.clicked.connect(self.increase_count)  # Connect to increase handler
        counter_layout.addWidget(increase_button)

        header_layout.addLayout(counter_layout)

        # Remove Button
        remove_button = QPushButton("X")
        remove_button.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;  /* Red */
                border-radius: 12px;
                color: white;
                font-size: 12px;
                padding: 5px;
                min-width: 30px;
                min-height: 30px;
                border: none;
            }
            QPushButton:hover {
                background-color: #FF5252;
            }
        """)
        remove_button.setFixedSize(30, 30)
        remove_button.clicked.connect(self.on_remove_clicked)  # Connect to the remove handler
        header_layout.addWidget(remove_button, alignment=Qt.AlignRight)

        main_layout.addLayout(header_layout)

        # Product Details
        details_label = QLabel(
            f"<b>Weight:</b> {self.product['weight']}<br>"
            f"<b>Info:</b> {self.product['info']}<br>"
            f"<b>Barcode:</b> {self.product['barcode']}"
        )
        details_label.setFont(QFont("Arial", 8))
        details_label.setStyleSheet("color: white; border: none;")  # No border
        details_label.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(details_label)

        self.setLayout(main_layout)

    def create_shadow(self):
        """Creates a subtle drop shadow effect."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(3)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 100))  # Soft black shadow
        return shadow

    def on_remove_clicked(self):
        """Handles the remove button click event."""
        # Emit the class_id of the product to remove
        self.remove_signal.emit(self.product['class_id'])

    def increase_count(self):
        """Increases the count of the product."""
        self.count += 1
        self.count_label.setText(str(self.count))
        # Emit the updated count
        self.count_changed_signal.emit(self.product['class_id'], self.count)

    def decrease_count(self):
        """Decreases the count of the product."""
        if self.count > 1:
            self.count -= 1
            self.count_label.setText(str(self.count))
            # Emit the updated count
            self.count_changed_signal.emit(self.product['class_id'], self.count)