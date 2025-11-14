"""import json
from model.search import IndexSearch 
from model.agent import QueryAgent
from model.tools import get_retrieval_tool, get_final_answer_tool
from prompts.query_cot import QUERY_COT_SYSTEM_PROMPT
import faiss, numpy as np, torch
from dotenv import load_dotenv
from const import get_embedding_model
load_dotenv()"""

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

def edge_to_nl(txt):
    """
    Convert a TKG edge into natural language using LLaMA.
    """

    prompt = TEXT_REPR_EXTRACTION_PROMPT.format(edge = txt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        temperature=0.1,
        eos_token_id=tokenizer.encode("\n")[0] #stops after one newline.
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text

print(edge_to_nl("Police_(South_Korea)	Mobilize_or_increase_police_power	South_Korea	2007-01-16"))