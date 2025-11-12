import faiss
import numpy as np
import json

from dotenv import load_dotenv
from const import get_embedding_model

load_dotenv()

class IndexSearch():
    
    def __init__(self, index_path, metadata_path, top_k = 10):
        
        self.index = faiss.read_index(index_path)
        self.model = get_embedding_model()
        self.top_k = top_k
        self.metadata = self.load_metadata(metadata_path)
        
    def load_metadata(self, metadata_path: str) -> dict:
        """
        Loads a metadata json into a python dictionary.
        
        Args:
            metadata_path (str): The path to the metadata json file.
        
        Returns:
            metadata (dict): The metadata file as a python dict.
        
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        return metadata
    
    def search(self, query: str) -> dict:
        """
        Performs a semantic similarity search on the index.
        
        Args:
            metadata_path (str): The path to the metadata json file.
        
        Returns:
            ret (dict): The dictionary containing ids and the corresponding text representaiton.
        """
        model = self.model
        
        embeddings = model.encode(query, convert_to_numpy=True, normalize_embeddings=True,)
        embeddings = np.array([embeddings]).astype("float32")
        
        _, ids = self.index.search(embeddings, self.top_k)
        
        retrieved_texts = [self.metadata[i]["text"] for i in ids[0]]
        retrieved_ids = [self.metadata[i]["id"] for i in ids[0]]
        
        ret = {retrieved_ids[i]: retrieved_texts[i] for i in range(self.top_k)}
        
        return ret