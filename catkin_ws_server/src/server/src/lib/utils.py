#!/usr/bin/python3
import random
import cv2
import rospy
from std_msgs.msg import String
import numpy as np

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # print(x, color, label)
    if label:
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def check_detect_cls_number(lists):
    count={}
    for i in lists:
        try: count[i] += 1
        except: count[i]=1
    return str(count)
    
def draw(img, boxes, model, confs_classes, color_list, FR_instance, FR_SW):
    label = check_detect_cls_number(confs_classes)
    s = img.shape[1] // 640
    
    h = s * 30
    label = label.lstrip("{")
    label = label.rstrip("}")
    label, _number = split_string_data(label)
    _label = sorted(label, reverse=True)
    final = img.copy()
    if FR_SW:
        from lib.Recognition.face import Face
        face_crop_size = 160
        for box, color, cls in zip(boxes, color_list, confs_classes):
            rate_w = 1 / (model[0] / img.shape[1])
            rate_h = 1 / (model[1] / img.shape[0])

            box[0] = box[0] * rate_w
            box[1] = box[1] * rate_h
            box[2] = box[2] * rate_w
            box[3] = box[3] * rate_h
            if int(box[0]) > 0 and int(box[0]) < final.shape[1] and \
                int(box[2]) > 0 and int(box[2]) < final.shape[1] and \
                int(box[1]) > 0 and int(box[1]) < final.shape[0] and \
                int(box[3]) > 0 and int(box[3]) < final.shape[0]:
                crop_face = final[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                face_image = cv2.resize(crop_face, (face_crop_size, face_crop_size), interpolation=cv2.INTER_CUBIC)
                label = FR_instance.FR_Verify(face_image)
            else:
                label = cls
                
            final = plot_one_box(box, img, color, label)

        for up_txt in _label:
            up_txt = up_txt.replace(" ", "")
            t_size = cv2.getTextSize(up_txt, 0, fontScale=2 / 3, thickness=3)[0]
            cv2.putText(final, up_txt, (30, h), 0, s, [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            h += t_size[1]+50

        return final
            
    else:
        for box, color, cls in zip(boxes, color_list, confs_classes):
            final = plot_one_box(box, img, color, cls)
            
        for up_txt in _label:
            up_txt = up_txt.replace(" ", "")
            t_size = cv2.getTextSize(up_txt, 0, fontScale=2 / 3, thickness=3)[0]
            cv2.putText(final, up_txt, (30, h), 0, s, [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            h += t_size[1]+50
            
        return final

def split_string_data(data):
    used_data = data.split(sep=',')
    length = len(used_data)
    return used_data, length