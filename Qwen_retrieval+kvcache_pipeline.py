#Pipeline completa con KV cache



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
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from transformers.cache_utils import DynamicCache

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

def build_chat_messages(context, question, image_path):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the context and the image."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"}
            ]
        }
    ]

def preprocess_with_cache(model, processor, messages):
    
    txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #print("\n[CONTEXT PROMPT SNIPPET]\n", txt[:300], "\n")
    img, vid = process_vision_info(messages)
   #enc = processor(text=[txt], images=img, videos=vid,return_tensors="pt", padding=True).to(model.device)
    enc = processor(
    text=[txt],
    images=img,
    videos=vid,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=4000 # <--------------------------------------MAX LENGHT CACHE
).to(model.device)

    with torch.no_grad():
        out = model(**enc, use_cache=True)

    kv_cache = out.past_key_values
    print("[KV LEN]", kv_cache[0][0].shape[-2]) 

    
    #full_ids = enc.input_ids                  
    #start_ids = full_ids[:, -1:].clone()    _____________

    start_text = "Answer:\n"
    start_ids = processor.tokenizer(start_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    
  
    return kv_cache, start_ids




def generate_with_cache(model, tokenizer, kv_cache, input_ids, max_new_tokens=64):
    generated = []
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    device = input_ids.device

    # Ricava la posizione iniziale dai token già presenti in cache
    start_pos = kv_cache.key_cache[0].shape[2]  # seq_len in cache
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
        position_ids += 1  # autoreg

    if not generated:
        return ""
    output_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)




def main(args):
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

            messages = build_chat_messages(context_text, q, img_path)
            kv_cache, start_ids = preprocess_with_cache(model, processor, messages)
            answer = generate_with_cache(model, tokenizer, kv_cache, start_ids)
            
            print("[QUESTION]",q)
            #print("[START IDS]", start_ids)
           #print("[START IDS TOKEN]", tokenizer.decode(start_ids[0]))
            print("[ANSWER]",answer)
            
            

            predictions.append({
                "question": q,
                "reference": ref,
                "answers": answer,
                "question_type": sample.get("question_type", "unknown")
            })

        except Exception as e:
            print(f"[{i}] Errore: {str(e)[:200]}")
            continue

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
    p.add_argument("--top_k", type=int, default=1)
    main(p.parse_args())

