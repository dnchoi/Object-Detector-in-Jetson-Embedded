# ONNX to Tensor RT 2021/07/21
---
## 🚨
> Convert ONNX to TensorRT
```bash
# if Tensor RT file is not exist 
cd Face-Edge
python3 main.py
```
main.py -> from lib import onnx2trt<br>
convert script 

```python
import tensorrt as trt

def add_camera_args(parser):
    parser.add_argument('--onnx', help='onnx file location inside ./lib/models')
    parser.add_argument('--fp', type=int, default=16, help='floating point precision. 16 or 32')
    return parser    

def convert_onnx_to_trt(onnx, fp):
    model = './models/onnx/{}.onnx'.format(onnx)
    output = './models/trt/{}_{}.trt'.format(onnx, fp)
    logger = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = []
    print('trt version', trt.__version__)
    if trt.__version__[0] >= '7':
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        if fp == 16:
            builder.fp16_mode = True

        with open(model, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            
        # reshape input from 32 to 1
        shape = list(network.get_input(0).shape)
        engine = builder.build_cuda_engine(network)
        with open(output, 'wb') as f:
            f.write(engine.serialize())
```

