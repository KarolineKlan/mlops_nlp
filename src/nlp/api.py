from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from pytorch_lightning import load_model
from transformers import AutoModel, AutoTokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global classifier, embedding, tokenizer, device, gen_kwargs
    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    embedding = AutoModel.from_pretrained("distilbert-base-uncased")
    classifier = load_model("models/SimpleModel.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding.to(device)
    # gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up")
    del embedding, tokenizer, device, classifier


app = FastAPI(lifespan=lifespan)


@app.post("/review/")
async def inference(input_string: str = "it was a bad movie"):
    """Infer semantics given a IMDB review."""
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)
    with torch.no_grad():
        bert_embedding = embedding(**inputs)
        output = classifier(bert_embedding.pooler_output)
    return output
