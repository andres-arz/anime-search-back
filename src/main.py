from fastapi import FastAPI
from src.model import Pic2Encoder, Pic2Tag, Pic2Scratch, Pic2Tag_v2

app = FastAPI()
app.include_router(Pic2Encoder.router,
                   prefix='/pic2encoder')

app.include_router(Pic2Tag.router,
                   prefix='/anime_pic2tag')

app.include_router(Pic2Tag_v2.router,
                   prefix='/anime_pic2tag_v2')

app.include_router(Pic2Scratch.router,
                   prefix='/pic2scratch')
