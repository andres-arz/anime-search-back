from fastapi import APIRouter
from fastapi import File

from src.model.util.nn import nn
import io
from starlette.responses import StreamingResponse

router = APIRouter()

# model_pic_encoder_resnet50_v4 tf.float16, 1000 classes, 30 epoch, 1500000 images recall 0.628
# model_pic_encoder_resnet50_v8 tf.float32, 8192 classes, 12 epoch, 3600000 images recall 0.422 Detect a lot of new
#                                                                                               classes per image
model = nn(pth_model='./src/model/tech/models/model_pic_encoder_resnet50_v4',
           name='Pic2Encoder_256')


@router.get('/', include_in_schema=False)
async def get():
    return model.get()


@router.get('/model', include_in_schema=False)
async def get_model():
    return model.getModel()


@router.post('/')
async def post(obj: bytes = File()):
    return StreamingResponse(io.BytesIO(model.inference(obj)))
