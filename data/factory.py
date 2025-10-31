#Factory class for creating temporal knowledge graph from multiTQ dataset.

import json
import pickle
import os
from typing import Dict

class MultiTQFactory():
    
    def __init__(self, data_dir="./MultiTQ"):
            self.data_dir = data_dir
            self.train_path = os.path.join(data_dir, "kg/train.txt")
            self.valid_path = os.path.join(data_dir, "kg/valid.txt")
            self.test_path = os.path.join(data_dir, "kg/test.txt")
        
            self.entity_mapping_path = os.path.join(data_dir, "kg/entity2id.json")
            self.relation_mapping_path = os.path.join(data_dir, "kg/relation2id.json")
            self.ts_mapping_path = os.path.join(data_dir, "kg/ts2id.json")
            
            self.entity_mapping = self.load_mapping_from_json(self.entity_mapping_path)
            self.relation_mapping = self.load_mapping_from_json(self.relation_mapping_path)
            self.ts_mapping = self.load_mapping_from_json(self.ts_mapping_path)
            
    def load_mapping_from_json(self, path:str) -> Dict:
        """
        Loads a mapping json to a dictionary.
        
        Parameters: 
            path (str): Path to mapping json.
        
        Returns: 
            mapping (Dict): Dictionary that represents mapping.
        """        
        with open(path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        return mapping
        4
    def build_temporal_knowledge_graph(self, path:str) -> Dict: 
        """
        Builds a temporal knowledge graph (TKG) from a text file.
        Args:
            path (str): Path to the text file containing the TKG data.

        Returns:
            graph (Dict[str, Set[Tuple[str, str, str, str]]]):
                {e1: ((e1, rel, e2, ts),...)}
        """
        graph = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                split_line = line.strip().split()
                
                e1 = split_line[0]
                rel = split_line[1]
                e2 = split_line[2]
                ts = split_line[3]
                
                if e1 not in graph:
                    graph[e1] = set()
                
                graph[e1].add((e1, rel, e2, ts))
        
        return graph

    def build_graphs(self, train_pkl_path, valid_pkl_path, test_pkl_path):
        """
        Builds temporal knowledge graphs for the train, validation, and test sets,
        then serializes them to pickle files.
        """
        train_graph = self.build_temporal_knowledge_graph(self.train_path)
        valid_graph = self.build_temporal_knowledge_graph(self.valid_path)
        test_graph = self.build_temporal_knowledge_graph(self.test_path)

        with open(train_pkl_path, "wb") as f:
            pickle.dump(train_graph, f)
        with open(valid_pkl_path, "wb") as f:
            pickle.dump(valid_graph, f)
        with open(test_pkl_path, "wb") as f:
            pickle.dump(test_graph, f)

                    
if __name__ == "__main__":
    with open("/home/shiyanl/COMP545/COMP545_project/data/MultiTQ/kg/tkbc_processed_data/test.pickle", "rb") as f:
        x = pickle.load(f)
        
        print(x)