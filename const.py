import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

#This file defines constants to bee used across the project.

_client = None
_embedding_model = None

def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
    return _client