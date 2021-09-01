import os
import sys
import numpy as np

# init FACE model
from lib.Recognition import face as face

class FACE_model():
    def __init__(self, padding=False):
        # other settings (maybe default)
        self.embedded_norm = False
        self.prec = 'fp16'  # int8
        self.flag = None
        self.no_display = False
        self.stream = True
        self.box_sizes = False

        self.no_detect = False
        self.detect_thread = False
        self.camera_thread = True
        self.sleep = 1
        self.count = 0

        self.padding_mode = padding
        self.padding = 0.2

    def Recog_init(self):  # for face recognition
        self.face_recognition   = face.Recognition(0)
        code_path = os.path.dirname(os.path.abspath(__file__))
        files_dir = os.path.join(code_path, 'face_recognition/FR_Datasets/commax/align')
        
        embedding_filename      = os.path.join(files_dir, "embeddings_dataset.txt")
        classname_filename      = os.path.join(files_dir, "class_names.txt")
        
        self.embeddings, self.labels    = self.face_recognition.load_data(embedding_filename)
        self.class_names                = self.face_recognition.load_classnames(classname_filename)

   