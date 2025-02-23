from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.models import initialize_model
from app.endpoints import get_router
import os

app = FastAPI()

# Initialize model and tokenizer
model, tokenizer = initialize_model()

# Register endpoints
app.include_router(get_router(model, tokenizer))

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
