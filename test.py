import json
from model.search import IndexSearch 
from model.agent import QueryAgent
from model.tools import get_retrieval_tool, get_final_answer_tool
from prompts.query_cot import QUERY_COT_SYSTEM_PROMPT
import faiss, numpy as np, torch
from dotenv import load_dotenv
from const import get_embedding_model
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts.text_repr_extraction import TEXT_REPR_EXTRACTION_PROMPT
import torch

LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

def edge_to_nl(edge):
    """
    Convert a TKG edge into natural language using LLaMA.
    """
    base = edge

    prompt = TEXT_REPR_EXTRACTION_PROMPT.format("edge", edge)


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        temperature=0.1
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return only the answer after "Answer:" for cleanliness
    if "Answer:" in text:
        text = text.split("Answer:")[1].strip()

    return text

print(edge_to_nl("head=Malaysia, tail=Association_of_Southeast_Asian_Nations, relation=Make_statement, ts=2007-01-15"))