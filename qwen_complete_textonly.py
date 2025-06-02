#DA SISTEMARE: prompt

import argparse
import json
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache
from torch.backends.cuda import SDPBackend
from torch.backends.cuda import flash_sdp_enabled
print("FlashAttention attivo:", flash_sdp_enabled())

#Questo per forzare un backend compatibile con la sliding window, altrimenti l'attenzione qudratica alloca satura la memoria a 3200 token



active_backends = {
    "FLASH_ATTENTION": torch.backends.cuda.flash_sdp_enabled(),
    "MATH": torch.backends.cuda.math_sdp_enabled(),
    "EFFICIENT_ATTENTION": torch.backends.cuda.mem_efficient_sdp_enabled(),

}



torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)



print("[BACKEND ATTIVO]")
for name, enabled in active_backends.items():
    print(f"- {name}: {'✅' if enabled else '❌'}")

print("[BACKEND FORZATI]")
print("- FLASH_ATTENTION:", torch.backends.cuda.flash_sdp_enabled())
print("- EFFICIENT_ATTENTION:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("- MATH:", torch.backends.cuda.math_sdp_enabled())


def load_clip_and_index(args):
    from transformers import AutoModel, CLIPImageProcessor

    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda:1").eval()
    clip_processor = CLIPImageProcessor.from_pretrained(args.retriever_path)
    index = faiss.read_index(os.path.join(args.index_path, "knn.index"))
    with open(os.path.join(args.index_json_path, "knn.json")) as f:
        index_map = json.load(f)
    with open(args.kb_wikipedia_path) as f:
        wiki = json.load(f)
    return clip_model, clip_processor, index, index_map, wiki

def extract_features(image_path, clip_model, clip_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.float16, device=clip_model.device)
    with torch.no_grad():
        features = clip_model.encode_image(pixel_values=pixel_values)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)

def retrieve_topk_pages(features, index, index_map, wiki, k):
    _, I = index.search(features, k)
    urls = [index_map[i][0] for i in I[0]]
    texts = ["\n".join(wiki[url]["section_texts"]) for url in urls]
    return texts

def build_chat_messages(context, question):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers shortly to questions based on the context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        }
    ]

def preprocess_with_cache(model, tokenizer, messages):
    prompt = "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>"
    enc = tokenizer(prompt,
                     return_tensors="pt",
                       padding=True,
                         truncation=True,
                           max_length=4000 #<------------------------------- MAX LENGHT
                           ).to(model.device)    

    num_total_tokens = enc.input_ids.shape[1]
    print(f"[TOKEN COUNTS] Totale: {num_total_tokens}")

    with torch.no_grad():
        out = model(**enc, use_cache=True)

    kv_cache = out.past_key_values
    print("[KV LEN]", kv_cache[0][0].shape[-2])

    start_text = "Answer:\n"
    start_ids = tokenizer(start_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    del prompt, enc, out
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    return kv_cache, start_ids

def generate_with_cache(model, tokenizer, kv_cache, input_ids, max_new_tokens=64):
    generated = []
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    device = input_ids.device

    start_pos = kv_cache.key_cache[0].shape[2]
    position_ids = torch.tensor([[start_pos]], device=device)

    for _ in range(max_new_tokens):
        out = model(
            input_ids=input_ids,
            past_key_values=kv_cache,
            position_ids=position_ids,
            use_cache=True
        )
        kv_cache = out.past_key_values
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits / 0.2, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        if next_tok.item() == eos:
            break

        generated.append(next_tok)
        input_ids = next_tok
        position_ids += 1

    if not generated:
        return ""
    output_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    ).to("cuda:0").eval()

    clip_model, clip_processor, index, index_map, wiki = load_clip_and_index(args)

    with open(args.input_path) as f:
        dataset = json.load(f)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    with open(args.output_path, "w") as outf:
        outf.write("[\n")

    for i, sample in enumerate(tqdm(dataset[:10], desc="Generating"), 1):
        q = sample["question"]
        img_path = sample.get("related_images")
        ref = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]
        kv_cache = DynamicCache()

        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Immagine non trovata: {img_path}")
            with Image.open(img_path) as im:
                im.verify()

            feats = extract_features(img_path, clip_model, clip_processor)
            ctxs = retrieve_topk_pages(feats, index, index_map, wiki, k=args.top_k)
            context_text = "\n\n".join(ctxs)

            messages = build_chat_messages(context_text, q)
            kv_cache, start_ids = preprocess_with_cache(model, tokenizer, messages)
            answer = generate_with_cache(model, tokenizer, kv_cache, start_ids)

            print("[QUESTION]", q)
            print("[ANSWER]", answer)

            used = torch.cuda.memory_allocated(device="cuda:0") / (1024 ** 2)
            print(f"[GPU MEM] After example {i}: {used:.2f} MiB")

            predictions.append({
                "question": q,
                "reference": ref,
                "answers": answer,
                "question_type": sample.get("question_type", "unknown")
            })

        except Exception as e:
            print(f"[{i}] Errore: {str(e)[:200]}")
            continue

        del feats, ctxs, context_text, messages, kv_cache, start_ids, answer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

        if i % 10 == 0 or i == len(dataset) or i == 1:
            with open(args.output_path, "a") as outf:
                for entry in predictions:
                    json.dump(entry, outf, ensure_ascii=False)
                    outf.write(",\n")
            predictions = []

    with open(args.output_path, "a") as outf:
        outf.write("\n]")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--retriever_path", required=True)
    p.add_argument("--index_path", required=True)
    p.add_argument("--index_json_path", required=True)
    p.add_argument("--kb_wikipedia_path", required=True)
    p.add_argument("--top_k", type=int, default=3)
    main(p.parse_args())



    """
    python qwen_complete_textonly.py \
    --input_path /leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/encyclopedic_vqa/cineca_dataloader_test.json \
    --output_path /leonardo/home/userexternal/fdibiase/qwen_generation_with_retrieval_images.json \
    --model_path /leonardo_scratch/large/userexternal/fdibiase/modelli/qwen2.5-3b-instruct \
    --retriever_path /leonardo_scratch/large/userexternal/fdibiase/modelli/EVA-CLIP-8B \
    --index_path /leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/index/visualRAG/encyclopedic_eva_clip/retrieval_index_image_noblack_l2 \
    --index_json_path /leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/index/visualRAG/encyclopedic_eva_clip/retrieval_index_json_image_l2 \
    --kb_wikipedia_path /leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/encyclopedic_kb_wiki.json
"""
