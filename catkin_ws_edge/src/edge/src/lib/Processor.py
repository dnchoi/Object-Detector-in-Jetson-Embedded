import cv2 
import sys
import os 
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import math
import time

class Processor():
    def __init__(self, model, anchor, outshape, class_number, 
                strides, model_input, fp_mode):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        print('trtbin : ', model)
        with open(model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({ 'host': host_mem, 'device': device_mem })
            else:
                outputs.append({ 'host': host_mem, 'device': device_mem })
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

        # post processing config
        self.output_shapes = outshape
        self.strides = np.array(strides)

        anchors = np.array(anchor)
        self._model_input = model_input
        if self._model_input[0] / 640 != 1:
            for i in range(3):
                lst = list(self.output_shapes[i])
                lst[2] = int(lst[2] * (self._model_input[0] / 640))
                lst[3] = int(lst[3] * (self._model_input[0] / 640))
                self.output_shapes[i] = tuple(lst)

        self.nl = len(anchors)
        self.nc = class_number # classes
        self.end_shape = self.nc + 5 # outputs per anchor
        self.na = len(anchors[0])
        self.fp = fp_mode
        if self.fp == 16:
            a = anchors.copy().astype(np.float16)
        elif self.fp == 32:
            a = anchors.copy().astype(np.float32)
        else:
            raise SystemExit('Error : floatting point')

        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

    def detect(self, img):
        resized = self.pre_process(img)
        outputs, fps = self.inference(resized)
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))
        return reshaped, fps

    def pre_process(self, img):
        # print('original image shape', img.shape)
        img = cv2.resize(img, tuple(self._model_input))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.fp == 16 or self.fp == 32:
            img = img.transpose((2, 0, 1)).astype(np.float32)
        else:
            raise SystemExit('Error : floatting point   2')

        img /= 255.0
        return img

    def inference(self, img):
        # copy img to input memory
        # self.inputs[0]['host'] = np.ascontiguousarray(img)
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        start = time.time()
        self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
       # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        end = time.time()
        # print('execution time:', 1/(end-start))
        return [out['host'] for out in self.outputs], 1/(end-start)

    def post_process(self, outputs, conf_thres=0.5):
        scaled = []
        grids = []
        for out in outputs:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor
            
            out = out.reshape((1, 3 * width * height, self.end_shape))

            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        return self.nms(pred)
    
    def make_grid(self, nx, ny):
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)
    def exponential_v(self, array):
        return np.exp(array)
    
    def non_max_suppression(self, boxes, confs, classes, iou_thres=0.6):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = confs.flatten().argsort()[::-1]
        keep = []
        # print(order.size, "\t", boxes)
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where( ovr <= iou_thres)[0]
            order = order[inds + 1]
        
        boxes = boxes[keep]
        confs = confs[keep]
        classes = classes[keep]
        # print(classes)
        return boxes, confs, classes

    def nms(self, pred, iou_thres=0.6):
        boxes = self.xywh2xyxy(pred[..., 0:4])
        # best class only
        confs = np.amax(pred[:, 5:], 1, keepdims=True)
        classes = np.argmax(pred[:, 5:], axis=-1)
        return self.non_max_suppression(boxes, confs, classes)

    def xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
