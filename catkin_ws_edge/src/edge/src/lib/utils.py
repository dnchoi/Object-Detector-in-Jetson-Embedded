#!/usr/bin/python3
import random
import cv2
import rospy
from std_msgs.msg import String
import json

def convert_bbox_resolution(bbox, model, cam):
    for i in range(len(bbox)):
        # print("model size : {}x{} camera size : {}x{}".format(model[0],model[1], cam[0], cam[1]))
        rate_w = 1 / (model[0] / cam[0])
        rate_h = 1 / (model[1] / cam[1])
        # print("rate XY : {}x{}".format(rate_w, rate_h))
        # print("Old bbox : {0:.3f}/{1:.3f}/{2:.3f}/{3:.3f}".format(bbox[0][0],bbox[0][1],bbox[0][2],bbox[0][3]))

        bbox[i][0] = bbox[i][0] * rate_w
        bbox[i][1] = bbox[i][1] * rate_h
        bbox[i][2] = bbox[i][2] * rate_w
        bbox[i][3] = bbox[i][3] * rate_h
        # print("New bbox : {0:.3f}/{1:.3f}/{2:.3f}/{3:.3f}".format(bbox[0][0],bbox[0][1],bbox[0][2],bbox[0][3]))
    return bbox

def gen_colors(classes):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]
    
    return colors

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def draw(img, boxes, confs, classes, color_list, train_name, pub, obj, node):
    final = img.copy()
    _x1 = ''
    _y1 = ''
    _x2 = ''
    _y2 = ''
    _confs = ''
    _cls = ''
    _r = ''
    _g = ''
    _b = ''
    i = 0
    for box, conf, cls in zip(boxes, confs, classes):
        conf = conf[0]
        cls_name = train_name[cls]
        _conf = '{0:.3f}'.format(conf)
        color = color_list[cls]
        # json_msg = {"number : {}, bbox : {}, text : {}, color : {}".format(i, box, _text, color)}
        # bbox_msg = "{0:.3f},{1:.3f},{2:.3f},{3:.3f}".format(box[0],box[1],box[2],box[3])
        # color_msg = "{0:},{1:},{2:}".format(color[0],color[1],color[2])
        # pub.publish(msgs)
        # print(msgs)
        
        final = plot_one_box(box, img, color, cls_name)
        _x1 += f"{round(box[0],3)},"
        _y1 += f"{round(box[1],3)},"
        _x2 += f"{round(box[2],3)},"
        _y2 += f"{round(box[3],3)},"
        _cls += f"{cls_name},"
        _confs += f"{_conf},"
        _r += f"{color[0]},"
        _g += f"{color[1]},"
        _b += f"{color[2]},"
        # _bboxs.append(boxes)
        # _confs.append(_text)
        # _colors.append(color)
        # print(_bboxs, _confs, _colors)
        i += 1 
    obj.Header.stamp = rospy.Time.now()
    obj.cls = _cls
    obj.conf = _confs
    obj.x1 = _x1
    obj.y1 = _y1
    obj.x2 = _x2
    obj.y2 = _y2
    obj.r = _r
    obj.g = _g
    obj.b = _b
    pub.publish(obj)

    return final
    # return _bboxs, _confs, _colors
