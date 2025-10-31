from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

class TKGNode():
    def __init__(self, entity):
        self.entity = entity
        self.edges = {}
        
    def add_edge(self, tuple):
        e1 = tuple[0]
        rel = tuple[1]
        e2 = tuple[2]
        ts = tuple[3]
        
        if e2 not in self.edges:
            self.edges[e2] = set()
            
            self.edges[e2].add(TKGEdge(e1, e2, rel, ts))


class TKGEdge():
    def __init__(self, head, tail, relation, ts):
        self.head_ent = head
        self.tail_ent = tail
        self.relation = relation
        self.time = ts
        
        self.processing_model = os.getenv("PROCESSING_MODEL")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.prompt_id = os.getenv("TEXT_REPR_EXTRACTION_PROMPT_ID")
        self.client = OpenAI()
        self.text_repr = self.generate_text_repr()
        self.embedding = self.generate_embedding()
    
    def generate_text_repr(self):
        
        input = f"head={self.head_ent}, tail={self.tail_ent}, relation={self.relation}, ts={self.time}"
        
        response = self.client.responses.create(
            model = self.processing_model,
            prompt = {
                "id": self.prompt_id
            },
            input = input
        )
        
        print(response.output[0].content[0].text)
        return response.output[0].content[0].text

    def generate_embedding(self):
        
        input = self.text_repr
        
        response = self.client.embeddings.create(
            model = self.embedding_model,
            input = input
        )
        
        return response.data[0].embedding
    
if __name__ == "__main__":
   node = TKGNode("Obama")
   
   t = ("Obama", "PresidentOf", "United States", 2009)
   node.add_edge(t)
   print(list(node.edges["United States"])[0].text_repr)