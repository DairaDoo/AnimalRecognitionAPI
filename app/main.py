from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image
import io, pickle
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# --- Habilitar CORS para tu HTML en Live Server ---
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Cargar índice y metadata ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index = faiss.read_index(os.path.join(BASE_DIR, "../index.faiss"))
with open(os.path.join(BASE_DIR, "../id_to_meta.pkl"), "rb") as f:
    id_to_meta = pickle.load(f)

# Path to images
IMAGE_FOLDER = r"C:\codigo_proyectos_localmente_estudio\EncuentraTuMascotaPR\EncuentraTuMascotaPRAPI\wwwroot\reports"

def image_to_embedding(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**{k: v.to(device) for k, v in inputs.items()})
    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype("float32")

@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve images from the reports folder"""
    image_path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    return FileResponse(image_path)

@app.post("/search")
async def search_image(file: UploadFile = File(...), top_k: int = Form(5)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al abrir la imagen: {str(e)}")
    
    emb = image_to_embedding(img)
    D, I = index.search(emb, top_k)
    
    results = []
    for idx, dist in zip(I[0], D[0]):
        meta = id_to_meta[idx]
        results.append({
            "filename": meta["filename"],
            "url": f"http://127.0.0.1:8000/images/{meta['filename']}",
            "score": float(dist)
        })
    
    return {"results": results}