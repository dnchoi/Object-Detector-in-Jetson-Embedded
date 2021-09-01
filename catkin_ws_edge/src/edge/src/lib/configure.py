import configparser
import ast
from os import setreuid

class configures:
    def __init__(self):
        self.config = configparser.ConfigParser()

    def get_configure(self, ini, run_model):
        self.config.read(ini)
        trt_bin = self.config.get(run_model, 'trt')
        onnx_bin = self.config.get(run_model, 'onnx')
        use_network = self.config.get(run_model, 'network')
        anchors = ast.literal_eval(self.config.get(run_model, 'anchor'))
        out_shape = ast.literal_eval(self.config.get(run_model, 'shape'))
        cls = self.config.get(run_model, 'num')
        strides = ast.literal_eval(self.config.get(run_model, 'stride'))
        input_size = onnx_bin.split(sep='.onnx')[0].split(sep='-')[1]
        return use_network, trt_bin, onnx_bin, anchors, out_shape, int(cls), strides, [int(input_size), int(input_size)]

