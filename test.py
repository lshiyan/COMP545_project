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
from const import get_llama_tokenizer_and_model
import torch

LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer, model = get_llama_tokenizer_and_model()

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
    answer = text[len(prompt):]
    return answer

print(edge_to_nl("Abdul_Kalam\tExpress_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)\tSocial_Worker_(India)\t2005-01-01\n"))