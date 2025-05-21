from transformers import AutoTokenizer, Gemma3ForCausalLM
from transformers.cache_utils import DynamicCache
import argparse
import json
import logging
import os
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, CLIPImageProcessor
import faiss
import numpy as np
import torch
from tqdm import tqdm

# Impostazioni PyTorch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
os.environ["PYTORCH_SDP_ATTENTION"] = "0"

# ======= FUNZIONI DI SUPPORTO =======

def load_clip_and_index(args):
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda:1").eval()

    clip_processor = CLIPImageProcessor.from_pretrained(args.retriever_path)
    index = faiss.read_index(os.path.join(args.index_path, 'knn.index'))

    with open(os.path.join(args.index_json_path, 'knn.json'), 'r') as f:
        index_map = json.load(f)

    with open(args.kb_wikipedia_path, 'r') as f:
        wiki = json.load(f)

    return clip_model, clip_processor, index, index_map, wiki


def extract_features(image_path, clip_model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.float16, device=clip_model.device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        features = clip_model.encode_image(pixel_values=pixel_values)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        del pixel_values
        torch.cuda.empty_cache()

    if torch.isnan(features).any() or torch.isinf(features).any():
        raise ValueError(f"Feature extraction failed: NaN or Inf in features for image {image_path}")

    return features.cpu().numpy().astype(np.float32)


def retrieve_topk_pages(features, index, index_map, wiki, k):
    D, I = index.search(features, k)
    urls = [index_map[i][0] for i in I[0]]
    texts = ["\n".join(wiki[url]["section_texts"]) for url in urls]
    if any(len(text.strip()) == 0 for text in texts):
        raise ValueError("Retrieved empty context text!")
    return texts


def preprocess_knowledge(model, tokenizer, prompt: str, kv_cache):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2000,   
        padding=False
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    print(f"[Prompt token count: {input_ids.shape[-1]}")

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            attn_implementation="eager",
            use_cache=True
        )
    start_ids = input_ids[:, -1:].clone()
    return kv_cache, start_ids



def generate_answer_with_cache(model, tokenizer, kv_cache, start_ids, max_new_tokens=64):
    device = model.device
    input_ids = start_ids.to(device)  
    generated = [input_ids]

    for step in range(max_new_tokens):
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                use_cache=True,
                attn_implementation="eager"
            )
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits / 0.2, dim=-1)  # temperatura 0.2
            next_token = torch.multinomial(probs, num_samples=1)  # sampling

            if next_token.item() == tokenizer.eos_token_id:
                print("[EOS token encountered]")
                break

            generated.append(next_token)
            input_ids = next_token  # shape [1, 1], per il passo successivo

    output_ids = torch.cat(generated[1:], dim=1)  # salta start_ids
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ======= MAIN =======

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_model_path, local_files_only=True)
    model = Gemma3ForCausalLM.from_pretrained(
        args.gemma_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True
    ).to("cuda:0").eval()

    clip_model, clip_processor, index, index_map, wiki = load_clip_and_index(args)

    with open(args.input_path, 'r') as f:
        dataset = json.load(f)

    predictions = []
    written_any = False
    log = logging.getLogger("bad-samples")
    log.setLevel(logging.INFO)
    log.addHandler(logging.FileHandler("bad_samples.log"))

    with open(args.output_path, "w") as f:
        f.write("[\n")

    for i, sample in enumerate(tqdm(dataset[:11], desc="Generazione"), 1):
        q = sample["question"]
        img = sample["related_images"]
        ref = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]
        kv_cache = DynamicCache() 

        try:
            if not os.path.exists(img):
                raise FileNotFoundError(f"Immagine non trovata: {img}")
            with Image.open(img) as image:
                image.verify()

            feats = extract_features(img, clip_model, clip_processor)
            ctxs = retrieve_topk_pages(feats, index, index_map, wiki, k=args.top_k)
            context_text = "\n\n".join(ctxs)
            messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers concisely using the given context."
    },
    {
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion:\n{q}\n\nAnswer:"
    }
]
     


            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if not full_prompt.startswith(tokenizer.bos_token):
                full_prompt = tokenizer.bos_token + full_prompt

            print("\n[DEBUG PROMPT]")
            print(full_prompt[:500])
            print("\n\n")
           

            kv_cache, start_ids = preprocess_knowledge(model, tokenizer, full_prompt, kv_cache)
            print(f"[{i}] Cache layers: {len(kv_cache.key_cache)}")
            for layer, layer_cache in enumerate(kv_cache.key_cache):             #
                print(f"  Layer {layer} -> keys: {layer_cache.size(-2)} tokens") #


            ans = generate_answer_with_cache(model, tokenizer, kv_cache, start_ids)

            predictions.append({
                "question": q,
                "reference": ref,
                "answers": ans,
                "question_type": sample.get("question_type", "unknown")
            })

        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            log.error(f"Errore nell'immagine {img}: {e}")
            continue
        except Exception as e:
            log.error(f"[{i}] Errore durante la generazione per {q}: {e}")
            continue

        if i % 100 == 0 or i in {1, 10, len(dataset)}:
            with open(args.output_path, "a") as f:
                for entry in predictions:
                    if written_any:
                        f.write(",\n")
                    json.dump(entry, f, ensure_ascii=False)
                    written_any = True
            predictions = []

    if written_any:
        with open(args.output_path, "a") as f:
            f.write("\n]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gemma_model_path", type=str, required=True)
    parser.add_argument("--retriever_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--index_json_path", type=str, required=True)
    parser.add_argument("--kb_wikipedia_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()
    main(args)
