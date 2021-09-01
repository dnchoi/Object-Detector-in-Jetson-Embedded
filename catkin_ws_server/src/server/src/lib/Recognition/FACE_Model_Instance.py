'''
Face Recognition
'''
from lib.Recognition.FACE_Model_Load import FACE_model

class FR:
    def __init__(self):
        self.Fz = FACE_model()
        self.Fz.Recog_init()

    def FR_Verify(self, input_faces):
        em, name, dis = self.Fz.face_recognition.verify(input_faces, self.Fz.embeddings, self.Fz.labels)
        label = self.Fz.class_names[int(name)]
        dis_s = "{0:.3f}".format(dis)
        total = f"{label}_{dis_s}"
        return total