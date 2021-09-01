#Model inits
import cv2
import sys
import argparse
import os 
from lib import utils
from lib import classes
from lib import capture_gstreamer
from lib import Subscriber
import time

#ROS inits
import rospy
from server.msg import ObjectDetection

class inference:
    def __init__(self):
        pass
    
    def check_classes(self, name):
        if name == "coco":
            what_cls = classes.coco
        elif name == "mask-face":
            what_cls = classes.mask_face
        if name == "wider-face":
            what_cls = classes.wider_face
        return what_cls

    def run(self, args, img, size):
        sub = Subscriber.subscriber()
        rospy.init_node('local')
        rospy.Subscriber("nx_pub", ObjectDetection, sub._nx_callback)
        rospy.Subscriber("nano_pub", ObjectDetection, sub._nano_callback)
        rospy.Subscriber("xavier_pub", ObjectDetection, sub._xavier_callback)

        # parse arguments
        # args = get_argparse()
        train_name = args.model.split(sep='.')[0].split(sep='_')[0]
        clss = self.check_classes(train_name)
        color_list = utils.gen_colors(clss)
        # model_input = [640, 480]

        prevTime = 0
        # cap = capture_gstreamer.Camera(args)
        cam_output = size
        # if not cap.isOpened():
        #     raise SystemExit('Error : Camera is not Open')
        # while cap.isOpened():
            # img = cap.read()
            # if args.cap_size:
            #     img = cv2.resize(img, dsize=(640, 480))
            # if img is None:
            #     break
        curTime = time.time(); detect_fps = 0
        outimg = img.copy()
        # # inference
        # print(cam_output)
        print(f"Nx -> {sub.nx_bbox} / {sub.nx_text} / {sub.nx_color}")
        print(f"Nano -> {sub.nano_bbox} / {sub.nano_text} / {sub.nano_color}")
        print(f"Xavier -> {sub.xavier_bbox} / {sub.xavier_text} / {sub.xavier_color}")
        if sub.nx_text is not None:
            new_nx_bbox = utils.convert_bbox_resolution(sub.nx_bbox, cam_output, cam_output)
            outimg = utils.draw(outimg, new_nx_bbox, sub.nx_text, sub.nx_color)
        if sub.nano_text is not None:
            new_nano_bbox = utils.convert_bbox_resolution(sub.nano_bbox, cam_output, cam_output)
            outimg = utils.draw(outimg, new_nano_bbox, sub.nano_text, sub.nano_color)
        if sub.xavier_text is not None:
            new_xavier_bbox = utils.convert_bbox_resolution(sub.xavier_bbox, cam_output, cam_output)
            outimg = utils.draw(outimg, new_xavier_bbox, sub.xavier_text, sub.xavier_color)
        
        now_time = time.time()
        print(now_time - sub.nx_cur_time)
        print(now_time - sub.nano_cur_time)
        print(now_time - sub.xavier_cur_time)
        if now_time - sub.nx_cur_time > 1:
            sub.nx_bbox = []
            sub.nx_text = None
            sub.nx_color = []
        elif now_time - sub.nano_cur_time > 1:
            sub.nano_bbox = []
            sub.nano_text = None
            sub.nano_color = []
        elif now_time - sub.xavier_cur_time > 1:
            sub.xavier_bbox = []
            sub.xavier_text = None
            sub.xavier_color = []
        # cv2.imshow("output", outimg)
        
        sec = curTime - prevTime
        prevTime = curTime
        print("Total FPS : {} / Detect FPS : {}".format(1/sec, detect_fps))
        
        return outimg

    