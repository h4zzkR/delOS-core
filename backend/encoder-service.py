from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from core.nlu.encoder.bert_encoder import SentenceEncoderService

# fastApi сервер для энкодера
# желательно запускать на хосте с CUDA

class Sequence(BaseModel):
    sequence: List[str]

app = FastAPI()
encoder = SentenceEncoderService()
encoder.featurize(["hotinit"]) # первый encode долгий

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/encode/")
def encode_sequence(sequence : Sequence):
    item_dict = sequence.dict()
    if sequence.sequence:
        out = encoder.featurize(sequence.sequence)
        item_dict.update({'encoded': out})
    return item_dict
