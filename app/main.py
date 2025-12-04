import os
import io
import pickle
import faiss
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
from typing import Annotated 

# Load environment variables
load_dotenv()

# --- Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "../index.faiss")
META_PATH = os.path.join(BASE_DIR, "../id_to_meta.pkl")
IMAGE_FOLDER = os.getenv("ImagePath")

# Global variables for models
ml_models = {}
device = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models and index on startup
    print("Loading CLIP Model...")
    ml_models["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    ml_models["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading Index...")
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        ml_models["index"] = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            ml_models["meta"] = pickle.load(f)
    else:
        print("WARNING: Index or Meta file not found. Search will fail.")
        ml_models["index"] = None
        ml_models["meta"] = None
        
    yield
    # Clean up resources if necessary
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_embedding(image: Image.Image) -> np.ndarray:
    processor = ml_models["processor"]
    model = ml_models["model"]
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**{k: v.to(device) for k, v in inputs.items()})
    
    # Normalize
    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype("float32")

@app.get("/images/{filename}")
async def serve_image(filename: str):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.post("/search")
async def search_image(file: UploadFile = File(...), top_k: Annotated[int, Query(ge=1, le=50)] = 5):
    if ml_models["index"] is None:
        raise HTTPException(status_code=500, detail="Index not loaded. Run prepare_index.py first.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    # Generate embedding
    query_emb = image_to_embedding(img)
    
    # Search
    # D = Distances (Dot Product scores), I = Indices
    D, I = ml_models["index"].search(query_emb, top_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1: continue # No result found by FAISS
        
        meta = ml_models["meta"].get(idx)
        if meta:
            results.append({
                "filename": meta["filename"],
                "url": meta["url"],
                "score": float(score) # Score close to 1.0 means identical
            })
    
    return {"results": results}