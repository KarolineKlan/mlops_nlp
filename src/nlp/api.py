from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from nlp.model import nlpModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global classifier, embedding, tokenizer, device, gen_kwargs
    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    embedding = AutoModel.from_pretrained("distilbert-base-uncased")
    classifier = nlpModel.load_from_checkpoint("models/SimpleModel.ckpt")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    embedding.to(device)
    classifier.to(device)
    # gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up")
    del embedding, tokenizer, device, classifier


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/review/")
async def inference(input_string: str = "it was a bad movie"):
    """Infer semantics given a IMDB review."""
    """Infer semantics given an IMDB review."""
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)
    with torch.no_grad():
        bert_output = embedding(**inputs)
        bert_embedding = bert_output.last_hidden_state[:, 0, :]  # Get the last hidden state
        # Pass the embeddings to the classifier

        output = classifier(bert_embedding)

        label = ["positive" if output > 0.5 else "negative"]
    return label[0], output.item()
