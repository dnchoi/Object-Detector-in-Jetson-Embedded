from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui
from PyQt5.QtGui import *
import time

class Full_display(QThread):
    signal = pyqtSignal(QtGui.QImage)
    
    def __init__(self, width, height, parent=None):
        super().__init__()
        self.Thread_SW = True
        self.gui_w = width
        self.gui_h = height
        self.full_img = None
        
    def run(self):
        while self.Thread_SW:
            if self.full_img is not None:
                self.signal.emit(self.full_img)