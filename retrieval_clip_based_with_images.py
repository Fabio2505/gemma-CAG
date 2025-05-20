"""
Multimodal retrieval‑augmented generation with Gemma‑3.
• Recupera contesto testuale con EVA‑CLIP + FAISS.
• Usa Gemma3ForConditionalGeneration in modalità chat‑template con immagine + testo.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from traceback import format_exc

import faiss
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPImageProcessor,
    Gemma3ForConditionalGeneration,
)



def load_clip_and_index(args):
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        
    ).to("cuda:1").eval()
    clip_processor = CLIPImageProcessor.from_pretrained(args.retriever_path)

    index = faiss.read_index(os.path.join(args.index_path, "knn.index"))
    with open(os.path.join(args.index_json_path, "knn.json")) as f:
        index_map = json.load(f)
    with open(args.kb_wikipedia_path) as f:
        wiki = json.load(f)

    return clip_model, clip_processor, index, index_map, wiki


def extract_features(image_path: str, clip_model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.float16, device=clip_model.device)

    with torch.no_grad():
        feats = clip_model.encode_image(pixel_values=pixel_values)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

    return feats.cpu().numpy().astype(np.float32)


def retrieve_text(features, index, index_map, wiki, k: int):
    _, I = index.search(features, k)
    urls = [index_map[i][0] for i in I[0]]
    texts = ["\n".join(wiki[url]["section_texts"]) for url in urls]
    return "\n\n".join(texts)


def build_chat_prompt(context: str, question: str, image: Image.Image):
    text_input = f"Context: {context}\n\nQuestion: {question}\n\nGive a short answer:"
    user_content = [{"type": "image", "image": image}, {"type": "text", "text": text_input}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_answer(model, processor, messages): 

    chat_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False  
)

    
    inputs = processor.tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=22000  # <----------------------------------------------------------------------- MAX TOKENS
)
    eos_ids = processor.tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    
    tokens = inputs["input_ids"][0].tolist()

    if tokens[-1] not in eos_ids:
        eos_token = torch.tensor([[eos_ids[0]]], device=inputs["input_ids"].device)
        attention = torch.tensor([[1]], device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], eos_token], dim=-1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], attention], dim=-1)

    # CUDA
    inputs = {
        k: v.to(model.device, dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to(model.device)
        for k, v in inputs.items()
    }


    #num_prompt_tokens = inputs["input_ids"].shape[-1]
    #print(f"[Prompt token count after truncation: {num_prompt_tokens}]") 


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.2,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    if gen_tokens.numel() == 0:
        raise ValueError("Nessun token generato")

    answer = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return answer.strip()


def main(args):
    processor = AutoProcessor.from_pretrained(args.gemma_model_path, local_files_only=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.gemma_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda:0").eval()

    clip_model, clip_proc, index, index_map, wiki = load_clip_and_index(args)

    with open(args.input_path) as f:
        dataset = json.load(f)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("bad-samples")
    log.setLevel(logging.INFO)
    log.addHandler(logging.FileHandler("bad_samples.log"))

    predictions, written_any = [], False
    with open(args.output_path, "w") as outf:
        outf.write("[\n")

    for i, sample in enumerate(tqdm(dataset, desc="Generating"), 1):
        q = sample["question"]
        img_path = sample.get("related_images", None)
        ref = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]

        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)
            with Image.open(img_path) as im:
                im.verify()

            feats = extract_features(img_path, clip_model, clip_proc)
            ctx = retrieve_text(feats, index, index_map, wiki, k=args.top_k)
            image = Image.open(img_path).convert("RGB")
            messages = build_chat_prompt(ctx, q, image)
            #print(messages)
            answer = generate_answer(model, processor, messages)
            #print(answer)
            
            predictions.append({
                "question": q,
                "reference": ref,
                "answers": answer,
                "question_type": sample.get("question_type", "unknown"),
            })
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            log.error(f"[IMG] {img_path}: {e}")
            continue
        except Exception as e:
            log.error(f"[{i}] {q}: {e}\n{format_exc()}")
            continue

        if i % 100 == 0 or i == len(dataset) or i==1:
            with open(args.output_path, "a") as outf:
                for entry in predictions:
                    if written_any:
                        outf.write(",\n")
                    json.dump(entry, outf, ensure_ascii=False)
                    written_any = True
            predictions = []

    with open(args.output_path, "a") as outf:
        if written_any:
            outf.write("\n]")
        else:
            outf.write("]")  # caso in cui non viene scritto nulla


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--gemma_model_path", required=True)
    p.add_argument("--retriever_path", required=True)
    p.add_argument("--index_path", required=True)
    p.add_argument("--index_json_path", required=True)
    p.add_argument("--kb_wikipedia_path", required=True)
    p.add_argument("--top_k", type=int, default=3) 
    main(p.parse_args())



