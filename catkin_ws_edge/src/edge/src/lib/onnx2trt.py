import tensorrt as trt

def convert_onnx_to_trt(onnx, trt_model, fp):
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

        with open(onnx, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            
        # reshape input from 32 to 1
        shape = list(network.get_input(0).shape)
        engine = builder.build_cuda_engine(network)
        with open(trt_model, 'wb') as f:
            f.write(engine.serialize())
        msg = """
        ---------------------------

        Convert ONNX to Tensor RT Finish
        writer is [luke.dn.choi]
        email - luke.dn.choi@funzin.co.kr

        ---------------------------
        """
        print(msg)

