"""
Classifications indicator window: a simple GUI that visualizes the results of EMG classification
according to the color displayed on the window.
"""

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


## Class for secondary window in which classifications can be visualized:
## (Inspired by example: https://www.pythonguis.com/tutorials/creating-multiple-windows/)
class ClassificationsWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it will appear as a free-floating window.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle('Classifications Indicator')
        self.setGeometry(100, 100, 800, 800)

        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor('grey'))
        self.setPalette(palette)

    def set_color(self, color='grey'):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)