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
        Generates embedding for edge.
        
        Args:
            embedding_model: The model to use for embedding.

        """
        embedding = embedding_model.encode(self.text_repr)
        self.embedding = torch.tensor(embedding, dtype=torch.float32)
    
class TKG(): 
    def __init__(self):
        self.edges = []
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
        
        Args:
            batch_size: Number of edges to embed at once.
        """
        if not self.pending_edges:
            return
        
        texts = [edge.text_repr for edge in self.pending_edges]
        
        embeddings = self.embedding_model.encode(texts, batch_size=batch_size)
        
        for edge, emb in zip(self.pending_edges, embeddings):
            edge.embedding = torch.tensor(emb, dtype=torch.float32)
        
        self.pending_edges.clear()
        
    def load_embedding_model(self) -> SentenceTransformer:
        """
        Loads embedding model.
        
        Returns: 
            model: The embedding model.
        """
        model_name = os.getenv("EMBEDDING_MODEL")
        model = SentenceTransformer(model_name)
        return model
        
    def build_faiss_index(self) -> None:
        """
        Builds a FAISS index on all edge embeddings.
        """
        if self.pending_edges:
            self.embed_edges()
        
        self.embedding_dim = self.edges[0].embedding.shape[0]
        
        # Convert embeddings to numpy array
        embeddings_array = np.array([edge.embedding.numpy() for edge in self.edges], dtype='float32')
        
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.faiss_index.add(embeddings_array)
    
    def query_index(self, query: str, k: int = 5) -> List[tuple]:
        """
        Query the FAISS index for similar edges.
        
        Args:
            query: The query string to search for.
            k: Number of top results to return.
        
        Returns:
            A list of tuples (edge, distance) for the top k most similar edges.
        """
        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.array([query_embedding], dtype='float32')
        
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