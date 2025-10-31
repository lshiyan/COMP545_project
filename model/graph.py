from dotenv import load_dotenv
from openai_client import get_openai_client
from typing import List
import os
import torch

load_dotenv()

class TKGNode():
    def __init__(self, entity):
        self.entity = entity
        self.edges = []
        
    def add_edge(self, relation:str, tail_entity:str, timestamp:str) -> None:
        """
        Adds an outgoing edge from this node to a tail entity.

        Args:
            relation (str): The relation type connecting this node to the tail entity.
            tail_entity (str): The entity that this edge points to.
            timestamp (str): The time associated with this edge.
        """
        edge = TKGEdge(self.entity, tail_entity, relation, timestamp)
        self.edges.append(edge)


class TKGEdge():
    def __init__(self, head, tail, relation, ts):
        self.head_ent = head
        self.tail_ent = tail
        self.relation = relation
        self.time = ts
        
        self.processing_model = os.getenv("PROCESSING_MODEL")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.prompt_id = os.getenv("TEXT_REPR_EXTRACTION_PROMPT_ID")
        self.text_repr = self.generate_text_repr()
        self.embedding = self.generate_embedding()
    
    def generate_text_repr(self) -> str:
        """
        Generates a textual representation of the edge.
        
        Returns:
            str: A natural language text description of the edge.
        """
        input = f"head={self.head_ent}, tail={self.tail_ent}, relation={self.relation}, ts={self.time}"
        
        client = get_openai_client()
        
        response = client.responses.create(
            model = self.processing_model,
            prompt = {
                "id": self.prompt_id
            },
            input = input
        )
        
        return response.output[0].content[0].text

    def generate_embedding(self) -> torch.Tensor:
        """
        Generates a vector embedding for the edge's textual representation.

        Returns:
            list[float]: The embedding.
        """
        input = self.text_repr
        client = get_openai_client()
        
        response = client.embeddings.create(
            model = self.embedding_model,
            input = input
        )
        
        embedding = response.data[0].embedding
        return torch.tensor(embedding, dtype=torch.float32)
    
class TKG(): 
    def __init__(self):
        self.nodes = {}
    
    def add_tuple(self, tuple: tuple) -> None:
        """
        Adds a tuple of the form (e1, rel, e2, ts) to the graph.

        Args:
            tuple (tuple): A tuple representing an edge in the graph.
        """
        
        e1 = tuple[0]
        rel = tuple[1]
        e2 = tuple[2]
        ts = tuple[3]
        
        if e1 not in self.nodes:
            self.nodes[e1] = TKGNode(e1)
        
        self.nodes[e1].add_edge(rel, e2, ts)