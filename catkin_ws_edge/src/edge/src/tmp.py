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
from python_qt_binding import loadUi
os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH": "/home/luke/.local/lib/python3.6/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"})
pyShow = None

#Model inits
import cv2
import argparse
from lib import Processor
from lib import utils
from lib import classes
from lib import capture_gstreamer
from lib import onnx2trt
from lib import configure
import time

#ROS inits
import rospy
from edge.msg import ObjectDetection

# Key input
import sys, select, termios, tty

def get_argparse():
    parser = argparse.ArgumentParser()
    parser = capture_gstreamer.add_camera_args(parser)
    parser.add_argument('--use_model', type=str, default=None, help='model name')
    parser.add_argument('--node', type=str, default=None, help='node name')
    parser.add_argument('--app', action='store_true', help='GUI used')
    parser.add_argument('--cap_size', action='store_true', help='resize image')
    
    args = parser.parse_args()
    return args

class pyMainwindow(QMainWindow):
    def __init__(self, opt):
        super(pyMainwindow, self).__init__()
        ui_dir = self.get_dir_path("ui/gui.ui")
        print(ui_dir)
        loadUi(ui_dir, self)
        self.init_parameters()
        self.initUI()
        # parse arguments
        self.args = opt
        rospy.init_node(self.args.node)
        self.msg_pub = rospy.Publisher(self.args.node+"_pub", ObjectDetection, queue_size=100)
        # parse arguments
        self.obj = ObjectDetection()
        self.obj.Header.frame_id = self.args.rtsp

        conf =  configure.configures()
        code_path = os.path.dirname(os.path.abspath(__file__))
        ini_file = os.path.join(code_path, 'config.ini')
        self.network, self.TRTbin, self.ONNXbin, self.anchors, self.out_shape, self.class_num, self.strides, self.model_input = conf.get_configure(ini_file, self.args.use_model)
        check_fp = self.TRTbin.split(sep='/')[6].find("16")
        train_name = self.TRTbin.split(sep='/')[6].split(sep='-')[0]
        # print(train_namqe)
        if check_fp > 0:
            self.fp = 16
        else:
            self.fp = 32

        if not os.path.isfile(self.TRTbin):
            onnx2trt.convert_onnx_to_trt(self.ONNXbin, self.TRTbin, self.fp)

        self.processor = Processor.Processor(self.TRTbin, self.anchors, self.out_shape, self.class_num, self.strides, self.model_input, self.fp)
        self.clss = self.check_classes(train_name)
        self.color_list = utils.gen_colors(self.clss)
            
        self.prevTime = 0
        self.cap = capture_gstreamer.Camera(self.args)
        self.cam_output = self.cap.get_size()
        print(self.cam_output)
        
        if not self.cap.isOpened():
            raise SystemExit('Error : Camera is not Open')
        
    def get_dir_path(self, file_name):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)
          
    def init_parameters(self):
        self.gray_color = "color : #000000; background-color : #4f4f4f; border-style: solid; border-width: 2px;    border-radius: 10px; "
        self.red_color = "color : #ffffff; background-color : #FF0000; border-style: solid; border-width: 2px;    border-radius: 10px; "
        self.green_color = "color : #ffffff; background-color : #00FF00; border-style: solid; border-width: 2px;    border-radius: 10px; "
        self.white_color = "color : #000000; background-color : #FFFFFF; border-style: solid; border-width: 2px;    border-radius: 10px; "

    def initUI(self):
        self.show_img.resize(640, 480)
        self.show_img.setStyleSheet(self.white_color)

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
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        elif e.key() == Qt.Key_F:
            self.showFullScreen()
        elif e.key() == Qt.Key_N:
            self.showNormal()
            
    def convertCvImage2QtImage(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(rgb_image).convert('RGB')
        return QtGui.QPixmap.fromImage(ImageQt(PIL_image))

    def Mat2QImage(self, label, img):
        img = cv2.resize(img, dsize=(640, 480))
        pixmap = self.convertCvImage2QtImage(img)
        label.setPixmap(pixmap)
        label.resize(pixmap.width(), pixmap.height())
        label.show()

    def check_classes(self, name):
        if name == "coco":
            what_cls = classes.coco
            return what_cls
        elif name == "maskface":
            what_cls = classes.maskface
            return what_cls
        if name == "widerface":
            what_cls = classes.widerface
            return what_cls

    def update_data(self):
        # while self.cap.isOpened():
        img = self.cap.read()
        
        if img is not None:
            if self.args.cap_size:
                img = cv2.resize(img, dsize=(640, 480))
            outimg = img.copy()
            # if img is None:
                # break
            self.cam_output = [img.shape[1], img.shape[0]]
            self.obj.Header.seq = self.cam_output[0]

            # print(self.cam_output)
            curTime = time.time(); detect_fps = 0
            output, detect_fps = self.processor.detect(img) 
            boxes, confs, classes = self.processor.post_process(output)
            if len(boxes) != 0:
                new_bbox = utils.convert_bbox_resolution(boxes, self.model_input, self.cam_output)
                outimg = utils.draw(img, new_bbox, confs, classes, self.color_list, self.clss, self.msg_pub, self.obj, self.args.node)
            else:
                self.obj.Header.stamp = rospy.Time.now()
                self.obj.cls_conf = ""
                self.obj.x1 = str(0)
                self.obj.y1 = str(0)
                self.obj.x2 = str(0)
                self.obj.y2 = str(0)
                self.obj.r = str(0)
                self.obj.g = str(0)
                self.obj.b = str(0)
                self.msg_pub.publish(self.obj)

            sec = curTime - self.prevTime
            self.prevTime = curTime
            # print("Total FPS : {} / Detect FPS : {}".format(1/sec, detect_fps))
            self.Mat2QImage(self.show_img, outimg)
    
class NO_GUI:
    def __init__(self, opt):
        self.args = opt
        rospy.init_node(self.args.node)
        self.msg_pub = rospy.Publisher(self.args.node+"_pub", ObjectDetection, queue_size=100)
        # parse arguments
        self.obj = ObjectDetection()
        self.obj.Header.frame_id = self.args.rtsp

        conf =  configure.configures()
        code_path = os.path.dirname(os.path.abspath(__file__))
        ini_file = os.path.join(code_path, 'config.ini')
        self.network, self.TRTbin, self.ONNXbin, self.anchors, self.out_shape, self.class_num, self.strides, self.model_input = conf.get_configure(ini_file, self.args.use_model)
        check_fp = self.TRTbin.split(sep='/')[6].find("16")
        train_name = self.TRTbin.split(sep='/')[6].split(sep='-')[0]
        # print(train_namqe)
        if check_fp > 0:
            self.fp = 16
        else:
            self.fp = 32

        if not os.path.isfile(self.TRTbin):
            onnx2trt.convert_onnx_to_trt(self.ONNXbin, self.TRTbin, self.fp)

        self.processor = Processor.Processor(self.TRTbin, self.anchors, self.out_shape, self.class_num, self.strides, self.model_input, self.fp)
        self.clss = self.check_classes(train_name)
        self.color_list = utils.gen_colors(self.clss)
            

        self.prevTime = 0
        self.cap = capture_gstreamer.Camera(self.args)
        self.cam_output = self.cap.get_size()
        
        if not self.cap.isOpened():
            raise SystemExit('Error : Camera is not Open')
        
    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def check_classes(self, name):
        if name == "coco":
            what_cls = classes.coco
            return what_cls
        elif name == "maskface":
            what_cls = classes.maskface
            return what_cls
        if name == "widerface":
            what_cls = classes.widerface
            return what_cls
        
    def main(self):
        while self.cap.isOpened():
            key = self.getKey()
            img = self.cap.read()
            if self.args.cap_size:
                img = cv2.resize(img, dsize=(640, 480))

            if img is None:
                break
            curTime = time.time(); detect_fps = 0
            # outimg = img.copy()
            # inference
            self.cam_output = [img.shape[1], img.shape[0]]

            
            output, detect_fps = self.processor.detect(img) 
            boxes, confs, classes = self.processor.post_process(output)
            if len(boxes) != 0:
                new_bbox = utils.convert_bbox_resolution(boxes, self.model_input, self.cam_output)
                outimg = utils.draw(img, new_bbox, confs, classes, self.color_list, self.clss, self.msg_pub, self.obj, self.args.node)
            # cv2.imshow("output", outimg)
            sec = curTime - self.prevTime
            self.prevTime = curTime
            # print("Total FPS : {} / Detect FPS : {}".format(1/sec, detect_fps))

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            if (key == '\x03'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    opt = get_argparse()
    print(opt)
    if opt.app:
        print("Use GUI")
        app = QApplication(sys.argv)
        pyShow = pyMainwindow(opt)
        timer = QTimer()
        timer.timeout.connect(pyShow.update_data)
        timer.start(33)
        pyShow.show()
        sys.exit(app.exec_())
    else:
        print("Not Use GUI")
        app = NO_GUI(opt)
        app.main()
    
if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)
    
    main()   

