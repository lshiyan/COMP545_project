from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from model.graph import TKG
import faiss
import time
import torch
import pickle

with open("data/MultiTQ/tkg/train_tkg.pkl", "rb") as pkl_f:
    tkg = torch.load(pkl_f, map_location=torch.device('cpu'), weights_only=False)

    print(len(tkg.embeddings))
"""with open("data/MultiTQ/tkg/train_tkg.pkl", "wb") as f:
    pickle.dump(tkg, f)
    
index = tkg.build_faiss_index()
faiss.index_gpu_to_cpu(index)
faiss.write_index(index, "data/MultiTQ/tkg/train_index.faiss")"""

