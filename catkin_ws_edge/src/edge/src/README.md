# Face Edge Project 2021/07/21
---
## 🚨 Models
> 🔎 [YoloV5](https://github.com/ultralytics/yolov5) - Detector
>> [models](./models)<br>
>> [lib](./lib)<br>

|Dataset|Model|Classes|mAP|Precision|Recall|Img Size|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|COCO-80class|v5s|person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow,'elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush'|36.7|-|-|640x640|
|Wider Face|v5xs|Face, None|65.58%|84.33%|60.00%|640x640|
|Wider Face|v5xs|Face, None|53.31%|79.68%|49.64%|416x416|
|Wider Face|v5xs|Face, None|44.64%|78.27%|41.50%|320x320|
|Face Mask|v5xs|Face, Mask|95.34%|95.61%|92.66%|640x640|
|Face Mask|v5xs|Face, Mask|95.07%|96.22%|91.03%|416x416|
|Face Mask|v5xs|Face, Mask|94.31%|95.15%|91.13%|320x320|
|COCO-4class|v5xs|Person, Car, Bus, Truck|58.53%|84.28%|53.51%|640x640|
|COCO-4class|v5xs|Person, Car, Bus, Truck|58.53%|84.28%|53.51%|416x416|
|COCO-4class|v5xs|Person, Car, Bus, Truck|58.53%|84.28%|53.51%|320x320|

---
## 📝 Argparse
```bash
app_edge.py [-h] [--image IMAGE] [--video VIDEO] [--video_looping]
                   [--rtsp RTSP] [--rtsp_latency RTSP_LATENCY] [--usb USB]
                   [--gstr GSTR] [--onboard ONBOARD] [--copy_frame]
                   [--do_resize] [--width WIDTH] [--height HEIGHT]
                   [--use_model USE_MODEL] [--node NODE] [--app] [--cap_size]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         image file name, e.g. dog.jpg
  --video VIDEO         video file name, e.g. traffic.mp4
  --video_looping       loop around the video file [False]
  --rtsp RTSP           RTSP H.264 stream, e.g.
                        rtsp://admin:123456@192.168.1.64:554
  --rtsp_latency RTSP_LATENCY
                        RTSP latency in ms [200]
  --usb USB             USB webcam device id (/dev/video?) [None]
  --gstr GSTR           GStreamer string [None]
  --onboard ONBOARD     Jetson onboard camera [None]
  --copy_frame          copy video frame internally [False]
  --do_resize           resize image/video [False]
  --width WIDTH         image width [640]
  --height HEIGHT       image height [480]
  --use_model USE_MODEL
                        model name
  --node NODE           node name
  --app                 GUI used
  --cap_size            resize image
```
---

---
## 💻 Convert ONNX to TensorRT
python3 main.py

if not exist *.trt file, run convert onnx to trt

---