#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from prompts.text_repr_extraction import TEXT_REPR_EXTRACTION_PROMPT
from dotenv import load_dotenv
from const import get_embedding_model, get_llama_tokenizer_and_model

load_dotenv()

def detect_gpu(request_gpu: bool) -> bool:
    return bool(request_gpu and torch.cuda.is_available())

def load_tsv(path: str) -> List[Tuple[str, str, str, str]]:
    """
    Loads a TSV file with exactly 4 columns per line:
    head, relation, tail, ts
    """
    facts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 4:
                raise ValueError(f"Invalid line (expected 4 columns): {line}")
            facts.append((cols[0], cols[1], cols[2], cols[3]))
    return facts

def build_texts(texts: List[str], tokenizer, model, batch_size: int = 4, cache_path: str = None) -> List[str]:
    """
    Converts edges into natural language sentences using a local HuggingFace Llama model.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    sentences = []
    new_cache = dict(cache)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prompts = [
            f"Instruction: {TEXT_REPR_EXTRACTION_PROMPT} \n head= {h}, relation= {r}, tail: {t}, ts: {ts}\n"
            for (h, r, t, ts) in batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs)

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print(generated_texts)
        for j, text in enumerate(generated_texts):
            # take only new continuation after the prompt if desired
            clean_text = text.strip()
            h, r, t, ts = batch[j]
            key = f"{h}|{r}|{t}|{ts}"
            sentences.append(clean_text)
            new_cache[key] = clean_text

        # incremental cache save
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(new_cache, f, ensure_ascii=False, indent=2)

        print(f"[Llama] Processed {i + len(batch)}/{len(texts)} edges")

    return sentences

def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    use_gpu: bool
) -> np.ndarray:
    """
    Returns float32 np.array of shape (N, D) with L2-normalized rows.
    """
    device = "cuda" if use_gpu else "cpu"
    # encode supports normalize_embeddings=True (cosine-friendly)
    # convert_to_numpy ensures we get np.ndarray directly
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    ).astype("float32")
    return embeddings

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool) -> faiss.Index:
    """
    Builds an IndexFlatIP for cosine-like search (embeddings are normalized).
    If GPU available & requested, constructs on CPU then moves to GPU for speed,
    but we always return a CPU index for persistence unless caller asks for GPU.
    """
    d = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(d)
    cpu_index.add(embeddings)

    # We will return CPU index for saving. If you want to keep a GPU copy
    # for searching in-process, move it to GPU separately after saving.
    if use_gpu:
        # Warm up GPU resources once (optional); not strictly needed for saving
        _ = faiss.StandardGpuResources()
    return cpu_index

def save_artifacts(
    outdir: str,
    index: faiss.Index,
    facts: List[Tuple[str, str, str, str]],
    save_embeddings: bool,
    embeddings: np.ndarray = None
):
    os.makedirs(outdir, exist_ok=True)

    # Save FAISS index
    index_path = os.path.join(outdir, "index.faiss")
    faiss.write_index(index, index_path)

    # Save metadata (ID-aligned with embeddings order: 0..N-1)
    metadata = [
        {"id": i, "head": h, "relation": r, "tail": t, "timestamp": ts,
         "text": f"head={h}, tail={t}, relation={r}, ts={ts}"}
        for i, (h, r, t, ts) in enumerate(facts)
    ]
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Optionally save the embedding matrix
    if save_embeddings and embeddings is not None:
        np.save(os.path.join(outdir, "embeddings.npy"), embeddings)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index over a Temporal Knowledge Graph TSV.")
    parser.add_argument("--input", required=True, help="Path to TKG TSV file (head, relation, tail, ts).")
    parser.add_argument("--outdir", required=True, help="Directory to write index and artifacts.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for embedding.")
    parser.add_argument("--save-embeddings", action="store_true", default=False, help="Also save embeddings.npy.")
    parser.add_argument("--use-gpu", action="store_true", default= False, help="Use GPU if available for embeddings/FAISS.")
    args = parser.parse_args()

    use_gpu = detect_gpu(args.use_gpu)
    if args.use_gpu and not use_gpu:
        print("[Info] --use-gpu was requested but CUDA not available. Falling back to CPU.")

    print("[1/5] Loading TKG TSV...")
    facts = load_tsv(args.input)
    if not facts:
        raise RuntimeError("No facts found in TSV.")

    print(f"[2/5] Loading embedding model ({os.getenv('EMBEDDING_MODEL') or 'all-MiniLM-L6-v2'})...")
    model = get_embedding_model()

    print(f"[3/5] Embedding {len(facts)} facts (batch_size={args.batch_size}, device={'cuda' if use_gpu else 'cpu'})...")
    texts = build_texts(facts)
    
    """embeddings = embed_texts(model, sentences, args.batch_size, use_gpu)  # L2-normalized float32

    print(f"[4/5] Building FAISS index (IndexFlatIP) over dim={embeddings.shape[1]}...")
    cpu_index = build_faiss_index(embeddings, use_gpu)

    print(f"[5/5] Saving artifacts to {args.outdir} ...")
    save_artifacts(args.outdir, cpu_index, facts, args.save_embeddings, embeddings)

    print("Done.")
    print(f" - Index:      {os.path.join(args.outdir, 'index.faiss')}")
    print(f" - Metadata:   {os.path.join(args.outdir, 'metadata.json')}")
    if args.save_embeddings:
        print(f" - Embeddings: {os.path.join(args.outdir, 'embeddings.npy')}")
    print(f"Index size: {cpu_index.ntotal}")"""

if __name__ == "__main__":
    tokenizer, model = get_llama_tokenizer_and_model()
    texts = [("Abdel_Fattah_Al-Sisi", "Head_of_Government_(Egypt)", "Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)", "2014-06-17")]
    sentences = build_texts(texts, tokenizer, model, batch_size = 16, cache_path = "data/kg/MultiTQ/test_cache.json")