import os
from PIL import Image
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle

# --- Configuración para leer imágenes del backend ASP.NET Core ---
IMAGE_FOLDER = r"C:\codigo_proyectos_localmente_estudio\EncuentraTuMascotaPR\EncuentraTuMascotaPRAPI\wwwroot\reports"
BASE_URL = "http://127.0.0.1:8000/images/"  # FastAPI will serve the images

device = "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_to_embedding(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**{k: v.to(device) for k, v in inputs.items()})
    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype("float32")

# --- Crear índice FAISS ---
d = 512
index = faiss.IndexFlatL2(d)
id_to_meta = {}

for i, filename in enumerate(os.listdir(IMAGE_FOLDER)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(path).convert("RGB")
        emb = image_to_embedding(img)
        index.add(emb)
        id_to_meta[i] = {"filename": filename, "url": f"{BASE_URL}{filename}"}

# Guardar índice y metadata
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faiss.write_index(index, os.path.join(ROOT_DIR, "index.faiss"))
with open(os.path.join(ROOT_DIR, "id_to_meta.pkl"), "wb") as f:
    pickle.dump(id_to_meta, f)

print(f"Índice creado con {len(id_to_meta)} imágenes del backend ASP.NET Core.")