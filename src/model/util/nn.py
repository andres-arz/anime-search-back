from src.model.tech import ModelONNX as onnx
import os
import requests
import shutil
from tqdm import tqdm


def download_file(pth_model):
    url = pth_model.split('/')[-1]
    url = 'https://models.arz.ai/' + url + '.onnx'

    re = requests.get(url, stream=True)

    file_size = int(re.headers.get('Content-Length', 0))

    with tqdm.wrapattr(re.raw, "read", total=file_size) as r:
        with open(pth_model + '.onnx', 'wb') as f:
            shutil.copyfileobj(r, f)


class nn:
    def __init__(self, pth_model, name):
        self.name = name
        if not os.path.exists(pth_model + '.onnx'):
            download_file(pth_model=pth_model)
        self.model = onnx(pth_model=pth_model + '.onnx', name=self.name)

    def get(self):
        return {
            'app': self.name,
            'model': self.model.get()}

    def getModel(self):
        return self.model.get_model()

    def inference(self, x):
        return self.model.forward(x)
