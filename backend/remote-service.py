from typing import List

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from core.nlu.encoder.bert_encoder import SentenceEncoderService


# fastApi сервер для энкодера
# желательно запускать на хосте с CUDA

class Sequence(BaseModel):
    sequence: List[str]


app = FastAPI()
encoder = SentenceEncoderService()
encoder.featurize(["hotinit"], just_embeddings=True)  # первый encode долгий


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/encoder_info/")
def encoder_info():
    return encoder.featurizer.get_sentence_embedding_dimension()

@app.post("/encode/")
def encode_sequence(sequence: Sequence, just_embeddings: Optional[str] = 'false'):
    just_embeddings = True if just_embeddings == 'true' else False
    item_dict = sequence.dict()
    if sequence.sequence:
        out = encoder.featurize(sequence.sequence, just_embeddings=just_embeddings)
        item_dict.update({'encoded': out})
    return item_dict
