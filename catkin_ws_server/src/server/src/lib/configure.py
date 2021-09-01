import configparser
import ast
from os import setreuid

class configures:
    def __init__(self):
        self.config = configparser.ConfigParser()
        
    def get_configure(self, ini, name):
        self.config.read(ini)

        sub_node = ast.literal_eval(self.config.get(name, 'sub_node'))
        face_recognition = ast.literal_eval(self.config.get(name, 'face_recognition'))
        return sub_node, face_recognition

