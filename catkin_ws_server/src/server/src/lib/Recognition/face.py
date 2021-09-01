# coding=utf-8
"""Face Recognition"""

import os
import numpy as np
import tensorflow as tf
import cv2
from lib.Recognition import facenet

#import detect
from sklearn.metrics.pairwise import cosine_distances

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
code_path = os.path.dirname(os.path.abspath(__file__))
files_dir = os.path.join(code_path, 'face_recognition/FR_ModelCheckpoints/20190528-175231-finetuning_vggface2')
facenet_model_checkpoint = os.path.join(files_dir, "20190528-175231.pb")
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

COSINE_THRESHOLD = 0.44

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.distance = None
        

    def area(self):
        w = self.bounding_box[2] - self.bounding_box[0]
        h = self.bounding_box[3] - self.bounding_box[1]
        return w*h


class Recognition:
    def __init__(self , gpu_num):
        self.encoder = Encoder()
        self.verifier = Verifier()
        
    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def load_classnames(self, fname):
        with open(fname) as f:
            data = f.read().splitlines()
        classnames = [x for x in data]
        return classnames

    def load_phones(self, fname):
        with open(fname) as f:
            data = f.read().splitlines()
        classnames = [x for x in data]
        return classnames

    def load_data(self, fname):
        with open(fname) as f:
            data = f.read().splitlines()
        data = [x.split(",") for x in data]
        labels = [x[0] for x in data]
        data = [[float(y) for y in x[1:]] for x in data]

        return np.array(data),np.array(labels)

    def verify(self, faces,  embeddings, labels):
        embedding = self.encoder.generate_embedding(faces)
        name, distance = self.verifier.verify(embedding, embeddings, labels)
        return embedding, name, distance
   
class Verifier:
    def __init__(self):
        pass

    def compareEuclideanEmbeddings(self, emb, embeddings):
        dist = np.sqrt(np.sum(np.square(np.subtract(emb, embeddings)),axis=1)) #1-N
        return dist

    def compareCosineEmbeddings(self, emb, embeddings):
        dist = cosine_distances(embeddings,[emb])
        return dist.flatten()

    def compareCosineEmbeddings_single(self, emb, embeddings):
        dist = cosine_distances([embeddings],[emb])
        return dist.flatten()

    def verify(self, embedding, embeddings, labels ,fz_test = False):
        dist = -1
        if embedding is not None:
            dists = self.compareCosineEmbeddings(embedding, embeddings)

            idx = np.argmin(dists)

            dist = dists[idx]
            if (dist < COSINE_THRESHOLD):
                return labels[idx], dist
        return "0",dist

class Encoder:
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        # with self.sess.as_default():
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]
 