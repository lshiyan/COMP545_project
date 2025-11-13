import torch
from transformers import pipeline

# ------------------------------
# 1️⃣ Model config
# ------------------------------
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# If you have 16GB+ VRAM (e.g., 3090, A100), you can use float16 or bfloat16
# Otherwise, set to "auto" to let Transformers decide
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Initialize pipeline — this automatically places weights on GPU
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch_dtype,
    device_map="auto",           # automatically uses all visible GPUs
    max_new_tokens=128,          # 512 is usually overkill for 1-sentence output
    temperature=0.3,
    do_sample=False,
)

# ------------------------------
# 2️⃣ Generation function
# ------------------------------
def edge_to_sentence(head, relation, tail, ts):
    """
    Converts a temporal knowledge-graph edge into a natural-language sentence.
    """
    prompt = (
        f"Convert the following to a natural language representation:\n"
        f"head = {head}\nrelation = {relation}\ntail = {tail}\ntimestamp = {ts}\n"
        f"Use fluent, grammatical English and express the date naturally."
    )

    result = pipe(prompt, batch_size=1)[0]["generated_text"]

    # remove the prompt echo
    if result.startswith(prompt):
        result = result[len(prompt):]
    return result.strip()

# ------------------------------
# 3️⃣ Example
# ------------------------------
if __name__ == "__main__":
    sentence = edge_to_sentence("Barack Obama", "BecamePresidentOf", "United States", "2009-01-05")
    print("\n[Llama Output] →", sentence)
    
"""from const import get_openai_client, get_llama_tokenizer_and_model
from model.query import COTExtractor
from model.search import IndexSearch
from prompts.text_repr_extraction import TEXT_REPR_EXTRACTION_PROMPT
import torch
tokenizer, model = get_llama_tokenizer_and_model()

h = "Abdul_Kalam"
r = "Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)"
t = "Social_Worker_(India)"
ts = '2005-01-01'

prompt = f"The quick brown fox"

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(generated_texts)"""
