from dotenv import load_dotenv
from openai_client import get_openai_client
from typing import List, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
import os

load_dotenv()

class TKGEdge():
    def __init__(self, head, tail, relation, ts):
        self.head_ent = head
        self.tail_ent = tail
        self.relation = relation
        self.time = ts
        self.text_repr = f"head={self.head_ent}, tail={self.tail_ent}, relation={self.relation}, ts={self.time}"
        self.embedding = None
        
    def generate_embedding(self, embedding_model: SentenceTransformer) -> None:
        """
        Generates embedding for edge. Computes on GPU, stores on CPU.
        
        Args:
            embedding_model: The model to use for embedding.
        """
        embedding = embedding_model.encode(self.text_repr, convert_to_tensor=True)
        # Move to CPU for storage
        self.embedding = embedding.cpu()
    
class TKG(): 
    def __init__(self, use_gpu = True):
        self.edges = []
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.embedding_model = self.load_embedding_model()
        self.pending_edges = [] #If any edges are added without embedding, they are stored here. When embed_edges is called, we batch embed all of these.
        
    def add_edge(self, tuple: tuple) -> None:
        """
        Adds an edge to the graph without batching.
        
        Args:
            tuple: A tuple (head, rel, tail, ts) representing an edge.
        """
        e1, rel, e2, ts = tuple
        edge = TKGEdge(e1, e2, rel, ts)
        self.pending_edges.append(edge)
        self.edges.append(edge)
    
    def add_and_embed_edge(self, tuple: tuple) -> None:
        """
        Adds an edge to the graph and embeds it immediately (single edge).
        Computes on GPU, stores on CPU.
        
        Args:
            tuple: A tuple (head, rel, tail, ts) representing an edge in this graph.
        """
        e1, rel, e2, ts = tuple
        edge = TKGEdge(e1, e2, rel, ts)
        edge.generate_embedding(self.embedding_model)
        self.edges.append(edge)
    
    def embed_edges(self, batch_size: int = 1024) -> None:
        """
        Generate embeddings for all pending edges in batches.
        Computes on GPU, stores on CPU.
        
        Args:
            batch_size: Number of edges to embed at once.
        """
        if not self.pending_edges:
            return
        
        texts = [edge.text_repr for edge in self.pending_edges]
        
        # Compute on GPU
        embeddings = self.embedding_model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_tensor=True,
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        # Store on CPU
        for edge, emb in zip(self.pending_edges, embeddings):
            edge.embedding = emb.cpu()
        
        self.pending_edges.clear()
        
    def load_embedding_model(self) -> SentenceTransformer:
        """
        Loads embedding model on GPU for computation.
        
        Returns: 
            model: The embedding model.
        """
        model_name = os.getenv("EMBEDDING_MODEL")
        device = "cuda" if self.use_gpu else "cpu"
        model = SentenceTransformer(model_name, device=device)
        return model
        
    def build_faiss_index(self) -> faiss.Index:
        """
        Builds a FAISS index on all edge embeddings.
        Uses GPU for computation if available, but embeddings are on CPU.
        
        Returns:
            A faiss index that consists of all edge embeddings of the TKG.
        """
        if self.pending_edges:
            self.embed_edges()
        
        self.embedding_dim = self.edges[0].embedding.shape[0]
        
        # Convert embeddings to numpy array (already on CPU)
        embeddings_array = np.array([edge.embedding.numpy() for edge in self.edges], dtype='float32')
        
        cpu_index = faiss.IndexFlatIP(self.embedding_dim)
        cpu_index.add(embeddings_array)
        
        # Use GPU FAISS for faster search, but data comes from CPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            faiss_index = cpu_index
        
        return faiss_index
    
    def query_index(self, query: str, k: int = 5) -> List[dict]:
        """
        Query the FAISS index for similar edges.
        Computes query embedding on GPU, searches on GPU FAISS index.
        
        Args:
            query: The query string to search for.
            k: Number of top results to return.
        
        Returns:
            A list of dicts containing the edge and textual representations of each edge.
        """
        # Compute query embedding on GPU
        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_tensor=True,
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        # Move to CPU and convert to numpy for FAISS
        query_embedding = query_embedding.cpu().numpy()
        query_embedding = np.array([query_embedding], dtype='float32')
        
        # Search (on GPU if FAISS index is on GPU)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            edge = self.edges[idx]
            results.append({
                'edge': edge,
                'text': edge.text_repr,
                'distance': float(distance)
            })
        
        return results