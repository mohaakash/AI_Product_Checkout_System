from PyQt5.QtWidgets import QPushButton, QGraphicsDropShadowEffect
from PyQt5.QtGui import QFont, QIcon, QColor
from PyQt5.QtCore import Qt, QSize

class CustomButton(QPushButton):
    def __init__(self, text="Button", color="#007BFF", hover_color="#0056b3", icon=None, on_click=None):
        super().__init__(text)

        # Set Font & Cursor
        self.setFont(QFont("Arial", 14, QFont.Bold))
        self.setCursor(Qt.PointingHandCursor)

        # Set Base Style
        self.color = color
        self.hover_color = hover_color
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color};
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                padding: 12px;
                min-width: 50px;
                max-width: 250px;
                min-height: 50px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self.hover_color};
            }}
            QPushButton:pressed {{
                background-color: {self.hover_color};
            }}
        """)

        # Set Icon if provided
        if icon:
            self.setIcon(QIcon(icon))
            self.setIconSize(QSize(32,32))  # Set initial icon size

        # Connect to function if provided
        if on_click:
            self.clicked.connect(on_click)

        # Add Drop Shadow
        self.setGraphicsEffect(self.create_shadow())

    def create_shadow(self):
        """Creates a drop shadow effect for the button."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)  # Shadow blur radius
        shadow.setXOffset(3)      # Horizontal shadow offset
        shadow.setYOffset(3)      # Vertical shadow offset
        shadow.setColor(QColor(0, 0, 0, 100))  # Shadow color (black with 100 alpha)
        return shadow

    def setIconSize(self, size):
        """Overrides setIconSize to make the icon bigger."""
        super().setIconSize(size * 1.5)  # Increase icon size by 1.5x