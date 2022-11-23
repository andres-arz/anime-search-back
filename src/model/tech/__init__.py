import onnxruntime as rt
import numpy as np
import io
import zlib


def compress_nparr(nparr):
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed


def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))


class ModelONNX:
    """
    Batch inference not supported
    """
    def __init__(self, name, pth_model):
        self.model_name = name
        providers = ['CPUExecutionProvider']
        self.model = rt.InferenceSession(pth_model, providers=providers)
        self.output_name = self.model.get_outputs()[0].name
        self.input_name = self.model.get_inputs()[0].name

    def forward(self, __x__):
        x = uncompress_nparr(__x__)
        onnx_pred = []
        for i in x:
            onnx_pred.append(self.model.run([self.output_name], {"input": i.astype(np.float32)})[0][0])

        return compress_nparr(np.array(onnx_pred).astype(np.float32))

    def get(self):
        return 'onnx run ' + self.model_name

    def get_model(self):
        return 'onnx' + self.model_name
