from fastapi import APIRouter
from fastapi import File

from src.model.util.nn import nn
import io
from starlette.responses import StreamingResponse
router = APIRouter()

model = nn(pth_model='./src/model/tech/models/model_anime_pic2tag_v4',
           name='Pic2Tag')


@router.get('/', include_in_schema=False)
async def get():
    return model.get()


@router.get('/model', include_in_schema=False)
async def get_model():
    return model.getModel()


@router.post('/')
async def post(obj: bytes = File()):
    return StreamingResponse(io.BytesIO(model.inference(obj)))
