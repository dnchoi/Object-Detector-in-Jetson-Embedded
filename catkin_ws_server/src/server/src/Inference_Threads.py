from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui
from PyQt5.QtGui import *
import cv2
from lib import utils
import time
import numpy as np

class Inference_worker(QThread):
    infer = pyqtSignal(list)
    
    def __init__(self, cap, callbacker, FR, width, height, full_size, parent=None):
        super().__init__()
        self.sub = callbacker
        self.cap = cap
        self.prevTime = 0
        self.Thread_SW = True
        self.FR_SW = FR
        self.gui_w = width
        self.gui_h = height
        self.FULL_SIZE = full_size
        if self.FR_SW:
            from lib.Recognition import FACE_Model_Instance
            self.FR_instance = FACE_Model_Instance.FR()
        else:
            self.FR_instance = None

    def run(self):
        while self.Thread_SW:
            if self.cap.is_opened:
                img = self.cap.read()
                infer_img = self._get_inference(img, self.sub._reso, self.sub._bbox, self.sub._color, self.sub._text, self.sub._cur_time)
                self.infer.emit(infer_img)

    def _get_inference(self, img, reso, bbox, color, text, curtime):
        if img is not None:
            tmp_img = img.copy()
            try:
                DIMENSIONS = {
                    "1920":(1920, 1080),
                    "1600":(1600, 1080),
                    "1280":(1280, 720),
                    "640":(640, 480),
                    "320":(320, 240),
                }
                new_width, new_height = DIMENSIONS[str(reso)]
                if int(reso) != tmp_img.shape[1]:
                    tmp_img = cv2.resize(tmp_img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                cam_output = [tmp_img.shape[1], tmp_img.shape[0]]
                tmp_img = utils.draw(tmp_img, bbox, cam_output, text, color, self.FR_instance, self.FR_SW)
    
                now_time = time.time()
                
                if now_time - curtime > 1 :
                    tmp_img = img.copy()
                    
                tmp_img2 = cv2.resize(tmp_img, dsize=(self.gui_w, self.gui_h), interpolation=cv2.INTER_CUBIC)
                full_width, full_height = DIMENSIONS[str(self.FULL_SIZE)]
                if int(self.FULL_SIZE) != tmp_img.shape[1]:
                    tmp_img = cv2.resize(tmp_img, dsize=(full_width, full_height), interpolation=cv2.INTER_CUBIC)

                return [tmp_img2, tmp_img]
            
            except Exception as inst:
                print(inst)
    