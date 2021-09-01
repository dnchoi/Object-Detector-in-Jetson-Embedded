#!/usr/bin/python3
# Qt APP inits
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PIL.ImageQt import ImageQt 
from PIL import Image
import os
import time
import numpy as np
os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH": "/home/luke/.local/lib/python3.6/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"})
pyShow = None

#Model inits
import cv2
import argparse
from lib import capture_gstreamer
from lib import Subscriber
from lib import configure
import Inference_Threads

#ROS inits
import rospy
from server.msg import ObjectDetection

class pyMainwindow(QMainWindow):
    def __init__(self):
        super(pyMainwindow, self).__init__()
        # ui_dir = self.get_dir_path("ui/gui.ui")
        # print(ui_dir)
        # loadUi(ui_dir, self)
        rospy.init_node('GPU_SERVER')
        self.args = self.get_argparse()
        
        conf =  configure.configures()
        code_path = os.path.dirname(os.path.abspath(__file__))
        ini_file = os.path.join(code_path, 'config.ini')
        self.node_names, self.FR = conf.get_configure(ini_file, "ros")
        self.initUI(self.node_names)
        self._init_CAM_ROS_SUB(self.node_names)
        
    def get_dir_path(self, file_name):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)
        
    def init_parameters(self):
        self.gray_color = "color : #000000; background-color : #4f4f4f; border-style: solid; border-width: 2px;    border-radius: 10px; "
        self.red_color = "color : #ffffff; background-color : #FF0000; border-style: solid; border-width: 2px;    border-radius: 10px; "
        self.green_color = "color : #ffffff; background-color : #00FF00; border-style: solid; border-width: 2px;    border-radius: 10px; "
        self.white_color = "color : #000000; background-color : #FFFFFF; border-style: solid; border-width: 2px;    border-radius: 10px; "
        
    def _init_CAM_ROS_SUB(self, node_names):
        self.sub = []
        self.cap = []
        self.work = []

        for n in range(len(node_names)):
            self.sub.append(Subscriber.subscriber())
            self.cap.append(None)
            self.work.append(None)
        
        for n in range(len(self.sub)):
            rospy.Subscriber(node_names[n], ObjectDetection, self.sub[n]._callback)
            
            while True:
                if self.sub[n]._addr is None:
                    print(f"Connect Capture cam {node_names[n]}")
                    time.sleep(1)
                    continue
                else:
                    print(self.sub[n]._addr)
                    self.args.rtsp = self.sub[n]._addr
                    self.cap[n] = capture_gstreamer.Camera(self.args)     
                    print(self.cap[n].is_opened)               
                    break
                            
            self.work[n] = Inference_Threads.Inference_worker(self.cap[n], self.sub[n], self.FR[n], self.GUI_width, self.GUI_height, self.FULL_GUI_WIDTH)
            if n == 0:
                updater = self.ui_update_0
            elif n == 1:
                updater = self.ui_update_1
            elif n == 2:
                updater = self.ui_update_2
                
            self.work[n].infer.connect(updater)
            self.work[n].start()
        
    def initUI(self, node):
        self.init_parameters()
        
        self.FULL_GUI_WIDTH, self.FULL_GUI_HEIGHT = 1600, 1080
        self.GUI_width, self.GUI_height = 320, 240
        Status_margen = 20
        mini_display_margen = 320
        node_length = len(node)
        self.status_text = "Ready"
        self.statusBar().showMessage(self.status_text)
        self.setGeometry(0, 0, self.FULL_GUI_WIDTH + mini_display_margen, self.FULL_GUI_HEIGHT + Status_margen)
        self.show_img = []
        
        self.pop_display = QLabel("FULL DISPLAY", self)
        self.pop_display.setAlignment(Qt.AlignCenter)
        self.pop_display.setGeometry(0, 0, self.FULL_GUI_WIDTH, self.FULL_GUI_HEIGHT)
        self.pop_display.setStyleSheet(self.white_color) 
        self.Area_widget = []
        for n in range(node_length):
            self.show_img.append(QLabel(f"{n}", self))

        for nn in range(len(self.show_img)):
            if nn==0:
                self.show_img[nn].setAlignment(Qt.AlignCenter)
                self.show_img[nn].setGeometry(self.FULL_GUI_WIDTH, 0, self.GUI_width, self.GUI_height)
                self.show_img[nn].setStyleSheet(self.white_color)
                self.Area_widget.append([self.FULL_GUI_WIDTH, 0])
            else:
                self.show_img[nn].setAlignment(Qt.AlignCenter)
                self.show_img[nn].setGeometry(self.FULL_GUI_WIDTH, nn * self.GUI_height, self.GUI_width, self.GUI_height)
                self.show_img[nn].setStyleSheet(self.white_color)
                self.Area_widget.append([self.FULL_GUI_WIDTH, nn * self.GUI_height])
            
        self.center()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_label_information(self, label_name, data):
        label_name.setText(data)

    def change_led_color(self, label_name, color):
        label_name.setStyleSheet(color)
        
    def loadImageFromFile(self, pixmap_label, img_path, rotation_angle) :
        self.load_data = QPixmap()
        self.load_data.load(img_path)
        self.load_data = self.load_data.scaled(pixmap_label.size(), Qt.KeepAspectRatio)
        rotation = QTransform().rotate(rotation_angle)
        self.load_data = self.load_data.transformed(rotation)
        pixmap_label.setPixmap(QtGui.QPixmap(self.load_data))            
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            self.close()
            for n in range(len(self.cap)):
                self.work[n].Thread_SW = False
                time.sleep(0.5)
                self.cap[n].release()
            sys.exit()
        elif e.key() == Qt.Key_F:
            self.showFullScreen()
        elif e.key() == Qt.Key_N:
            self.showNormal()
        
    def Mat2QImage(self, label, img, isFull):
        if isFull:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                h,w,c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                label.setPixmap(pixmap)
                label.show()
            except:
                print("Full error")
        else:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                h,w,c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                label.setPixmap(pixmap)
                label.show()
            except:
                print("error")
                
    def get_argparse(self):
        parser = argparse.ArgumentParser()
        parser = capture_gstreamer.add_camera_args(parser)
        parser.add_argument('--model', default='mask-face_16.trt', help='trt engine file located in ./models', required=False)

        args = parser.parse_args()
        return args

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.mouse_pt = [event.x(), event.y()]
    
    @pyqtSlot(list)
    def ui_update_0(self, img):
        try:
            num = 0
            self.Mat2QImage(self.show_img[num], img[0], False)
            if self.Area_widget[num][0] < self.mouse_pt[0] and self.Area_widget[num][0] + self.GUI_width > self.mouse_pt[0] and \
                self.Area_widget[num][1] < self.mouse_pt[1] and self.Area_widget[num][1] + self.GUI_height > self.mouse_pt[1]:
                self.full_img = img[1]
                self.statusBar().showMessage('CAM 1')
        except:
            pass
        
    @pyqtSlot(list)
    def ui_update_1(self, img):
        try:
            num = 1
            self.Mat2QImage(self.show_img[num], img[0], False)
            if self.Area_widget[num][0] < self.mouse_pt[0] and self.Area_widget[num][0] + self.GUI_width > self.mouse_pt[0] and \
                self.Area_widget[num][1] < self.mouse_pt[1] and self.Area_widget[num][1] + self.GUI_height > self.mouse_pt[1]:
                self.full_img = img[1]
                self.statusBar().showMessage('CAM 2')
                
        except:
            pass
        
    @pyqtSlot(list)
    def ui_update_2(self, img):
        try:
            num = 2
            self.Mat2QImage(self.show_img[num], img[0], False)
            if self.Area_widget[num][0] < self.mouse_pt[0] and self.Area_widget[num][0] + self.GUI_width > self.mouse_pt[0] and \
                self.Area_widget[num][1] < self.mouse_pt[1] and self.Area_widget[num][1] + self.GUI_height > self.mouse_pt[1]:
                self.full_img = img[1]
                self.statusBar().showMessage('CAM 3')
        except:
            pass
    
    def FULL_DISPLAY(self):
        try:
            self.Mat2QImage(self.pop_display, self.full_img, True)
        except:
            pass
    
def main():
    app=QApplication(sys.argv)
    pyShow = pyMainwindow()
    timer = QTimer()
    timer.timeout.connect(pyShow.FULL_DISPLAY)
    timer.start(33)

    pyShow.setWindowTitle("GPU Server")
    pyShow.show()
        
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()   

