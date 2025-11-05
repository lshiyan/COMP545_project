from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

def pad_attention_mask(mask, p):
    print(mask)
    B = mask.size(0)
    pad_tensor = torch.zeros(B, p)
    cat_tensor = torch.cat([pad_tensor, mask], dim=1)
    return cat_tensor.to(mask.dtype)

t = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0]])
    
print(pad_attention_mask(t, 3))