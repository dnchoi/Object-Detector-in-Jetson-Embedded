#!/usr/bin/python3

import cv2
import sys
import argparse
import os 
from lib import Processor
from lib import utils
from lib import classes
from lib import capture_gstreamer
from lib import onnx2trt
from lib import configure
import time

# #ROS init
import rospy
from edge.msg import ObjectDetection

# Key input
import sys, select, termios, tty

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def get_argparse():
    parser = argparse.ArgumentParser()
    parser = capture_gstreamer.add_camera_args(parser)
    parser.add_argument('--use_model', type=str, default=None, help='model name')
    parser.add_argument('--node', type=str, default=None, help='node name')
    args = parser.parse_args()
    return args

def check_classes(name):
    if name == "coco":
        what_cls = classes.coco
        return what_cls
    elif name == "maskface":
        what_cls = classes.maskface
        return what_cls
    if name == "widerface":
        what_cls = classes.widerface
        return what_cls

def main():
    args = get_argparse()
    rospy.init_node(args.node)
    msg_pub = rospy.Publisher(args.node+"_pub", ObjectDetection, queue_size=100)
    obj = ObjectDetection()
    # parse arguments
    conf =  configure.configures()
    code_path = os.path.dirname(os.path.abspath(__file__))
    ini_file = os.path.join(code_path, 'config.ini')
    network, TRTbin, ONNXbin, anchors, out_shape, class_num, strides, model_input = conf.get_configure(ini_file, args.use_model)
    check_fp = TRTbin.split(sep='/')[6].find("16")
    train_name = TRTbin.split(sep='/')[6].split(sep='-')[0]
    # print(train_namqe)
    if check_fp > 0:
        fp = 16
    else:
        fp = 32

    if not os.path.isfile(TRTbin):
        onnx2trt.convert_onnx_to_trt(ONNXbin, TRTbin, fp)

    processor = Processor.Processor(TRTbin, anchors, out_shape, class_num, strides, model_input, fp)
    clss = check_classes(train_name)
    color_list = utils.gen_colors(clss)
    
    prevTime = 0
    cap = capture_gstreamer.Camera(args)
    cam_output = cap.get_size()
    print(cam_output, model_input)
    if not cap.isOpened():
        raise SystemExit('Error : Camera is not Open')
    
    while cap.isOpened():
        key = getKey()
        img = cap.read()
        if img is None:
            break
        curTime = time.time(); detect_fps = 0
        # outimg = img.copy()
        # inference
        output, detect_fps = processor.detect(img) 
        boxes, confs, classes = processor.post_process(output)
        if len(boxes) != 0:
            new_bbox = utils.convert_bbox_resolution(boxes, model_input, cam_output)
            outimg = utils.draw(img, new_bbox, confs, classes, color_list, clss, msg_pub, obj, args.node)
        # cv2.imshow("output", outimg)
        sec = curTime - prevTime
        prevTime = curTime
        print("Total FPS : {} / Detect FPS : {}".format(1/sec, detect_fps))

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        if (key == '\x03'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)

    main()   

