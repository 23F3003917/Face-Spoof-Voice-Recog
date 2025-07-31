import os
import numpy as np

EMBEDDING_DIR = "embeddings"

def save_embedding(user_name, embedding):
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    np.save(os.path.join(EMBEDDING_DIR, f"{user_name}.npy"), embedding)

def load_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDING_DIR):
        if file.endswith(".npy"):
            name = file[:-4]
            path = os.path.join(EMBEDDING_DIR, file)
            embeddings[name] = np.load(path)
    return embeddings
