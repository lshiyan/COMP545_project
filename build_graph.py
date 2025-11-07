from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from model.graph import TKG
import time
import torch
import pickle

tkg = TKG()

with open("data/MultiTQ/kg/train.pkl", "rb") as pkl_f:
    train_edges = pickle.load(pkl_f)
    
    for entity in train_edges:
        edges = train_edges[entity]
        start_time = time.time()
        for edge in edges:
            tkg.add_edge(edge)
        tkg.embed_edges()
        print("Edges for entity " + entity + "loaded and embedded in " + str(time.time() - start_time) + "seconds.")