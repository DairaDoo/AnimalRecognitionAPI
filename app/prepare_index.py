import os
import pickle
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
IMAGE_FOLDER = os.getenv("ImagePath")
BASE_URL = "http://127.0.0.1:8000/images/"
BATCH_SIZE = 32 # Process images in chunks to speed up CPU usage
DEVICE = "cpu"

print("Loading model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_files(folder):
    # CRITICAL: Sort the files to ensure consistent ID mapping
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder) 
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

def create_embeddings(image_paths):
    all_embeddings = []
    total = len(image_paths)
    
    for i in range(0, total, BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        valid_indices = []

        # Load images safely
        for idx, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping bad image {path}: {e}")

        if not images:
            continue

        # Process batch
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.get_image_features(**{k: v.to(DEVICE) for k, v in inputs.items()})
        
        # Normalize embeddings (Critical for Cosine Similarity)
        # L2 Norm: vector / magnitude
        emb = outputs.cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        all_embeddings.append(emb)
        
        print(f"Processed {min(i + BATCH_SIZE, total)}/{total} images")

    if not all_embeddings:
        return np.empty((0, 512), dtype="float32")
        
    return np.vstack(all_embeddings)

def main():
    image_paths = get_image_files(IMAGE_FOLDER)
    
    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images. Generatings embeddings...")
    embeddings = create_embeddings(image_paths)
    
    # --- Create FAISS Index ---
    # We use IndexFlatIP (Inner Product) because our vectors are normalized.
    # This effectively calculates Cosine Similarity.
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings)

    # --- Create Metadata Map ---
    # Since we sorted image_paths, index ID 0 corresponds to image_paths[0]
    id_to_meta = {}
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        id_to_meta[i] = {
            "filename": filename,
            "url": f"{BASE_URL}{filename}"
        }

    # --- Save ---
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(ROOT_DIR, "index.faiss")
    meta_path = os.path.join(ROOT_DIR, "id_to_meta.pkl")

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(id_to_meta, f)

    print("------------------------------------------------")
    print(f"SUCCESS! Index saved to: {index_path}")
    print(f"Metadata saved to: {meta_path}")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()