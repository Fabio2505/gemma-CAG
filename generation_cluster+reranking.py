import argparse
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, CLIPImageProcessor, CLIPTokenizer
from torch.backends.cuda import flash_sdp_enabled

print("FlashAttention attivo:", flash_sdp_enabled())

def load_index_and_mapping(index_path, index_json_path):
    index = faiss.read_index(os.path.join(index_path, "knn.index"))
    with open(os.path.join(index_json_path, "knn.json")) as f:
        idx_map = json.load(f)
    return index, idx_map

def get_top_k_by_index(query_feat, documents, idx_map, faiss_index, k=10):
    url2pos = {}
    for pos, entry in enumerate(idx_map):
        url2pos.setdefault(entry[0], []).append(pos)
    sims = []
    for url in documents:
        for p in url2pos.get(url, []):
            vec = faiss_index.reconstruct(p)
            vec = vec / np.linalg.norm(vec)
            dist = float(1.0 - np.dot(query_feat.flatten(), vec.flatten()))
            sims.append((url, p, dist))
    sims.sort(key=lambda x: x[2])
    return [url for url,_,_ in sims[:k]]


def extract_features(image_path, question, clip_model, clip_processor, clip_tokenizer, alpha=0.5):

    # Carica e processa immagine
    image = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(dtype=torch.float16, device=clip_model.device)

    # Processa testo
   # txt_inputs = clip_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
   # input_ids = txt_inputs["input_ids"].to(device=clip_model.device)

    with torch.no_grad():
        image_feat = clip_model.encode_image(pixel_values)
        image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)

        #text_feat = clip_model.encode_text(input_ids)
        #text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)

        # Fusione lineare e normalizzazione finale
        #fused = alpha * image_feat + (1 - alpha) * text_feat
        #fused = fused / fused.norm(p=2, dim=-1, keepdim=True)

    return image_feat.cpu().numpy().astype(np.float32)


def cosine_distance(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1.0 - np.dot(a, b))


def retrieve_cluster_documents(feats, clusters_metadata_path):
    with open(clusters_metadata_path) as f:
        clusters = json.load(f)

    # appiattisci le feats in caso venissero come (1, D)
    q = feats.flatten()

    min_dist = float("inf")
    best_cluster = None

    for cid, data in clusters.items():
        # il centroide è già una lista di float di lunghezza D
        centroid = np.array(data["centroid"], dtype=np.float32)
        # calcola la cos distance
        dist = cosine_distance(q, centroid)
        if dist < min_dist:
            min_dist = dist
            best_cluster = cid

    # restituisci la lista di URL così com’è nel JSON
    print(f"[DEBUG] cluster {best_cluster} selected, min cosine distance = {min_dist:.4f}")

    return clusters[best_cluster]["documents"]




def build_chat_messages(context, question):
    return [
        {"role": "system", "content": "You are a helpful assistant that answers very shortly to questions based on the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"}
    ]

def preprocess_with_cache(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=20000).to(model.device)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    kv_cache = out.past_key_values
    start_text = "Answer:\n"
    start_ids = tokenizer(start_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    return kv_cache, start_ids

def generate_with_cache(model, tokenizer, kv_cache, input_ids, max_new_tokens=64):
    generated = []
    eos = tokenizer.eos_token_id
    device = input_ids.device
    start_pos = kv_cache.key_cache[0].shape[2]
    position_ids = torch.tensor([[start_pos]], device=device)
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, past_key_values=kv_cache, position_ids=position_ids, use_cache=True)
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
        local_files_only=True,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"}
    ).eval()

    clip_model = AutoModel.from_pretrained(args.retriever_path, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:1").eval()
    clip_processor = CLIPImageProcessor.from_pretrained(args.retriever_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.retriever_path)

    with open(args.input_path) as f:
        dataset = json.load(f)
    with open(args.kb_wikipedia_path) as f:
        wiki = json.load(f)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    first = True
    with open(args.output_path, "w") as outf:
        outf.write("[\n")

    for i, sample in enumerate(tqdm(dataset), 1):
        q = sample["question"]
        img_paths = sample.get("related_images")
        if isinstance(img_paths, list) and len(img_paths) > 0:
            img_path = img_paths[0]
        elif isinstance(img_paths, str):
            img_path = img_paths
        else:
            print(f"[{i}] Errore: Nessuna immagine valida associata alla domanda")
            continue

        ref = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]

        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Immagine non trovata: {img_path}")
            with Image.open(img_path) as im:
                im.verify()

            feats = extract_features(img_path, sample["question"], clip_model, clip_processor, clip_tokenizer)
            # 1) recupera il cluster intero
            doc_urls = retrieve_cluster_documents(feats, args.clusters_metadata_path)

            # 2) re-ranking con FAISS: prendi solo i top_k URL
            index, idx_map = load_index_and_mapping(args.index_path, args.index_json_path)
            top_docs = get_top_k_by_index(
                feats,            # feature query
                doc_urls,         # tutti gli URL del cluster
                idx_map,          # mapping [url, title, path]
                index,            # FAISS index
                k=args.top_k     # quanti tenere
            )
            print(f"[INFO] Top-{args.top_k} dopo re-ranking: {top_docs}")
            doc_urls = top_docs
            pages = [wiki[u] for u in doc_urls if isinstance(u, str) and u in wiki]

            context = "\n\n".join("\n".join(p["section_texts"]) for p in pages)
            messages = build_chat_messages(context, q)
            kv_cache, start_ids = preprocess_with_cache(model, tokenizer, messages)
            answer = generate_with_cache(model, tokenizer, kv_cache, start_ids)
            torch.cuda.empty_cache()

            print(f"[QUESTION] {q}\n[ANSWER] {answer}\n[REFERENCE] {ref}")

            with open(args.output_path, "a") as outf:
                if not first:
                    outf.write(",\n")
                json.dump({
                    "question": q,
                    "reference": ref,
                    "answers": answer,
                    "question_type": sample.get("question_type", "unknown")
                }, outf, ensure_ascii=False)
                first = False

        except Exception as e:
            import traceback
            print(f"[{i}] Errore: {str(e)}")
            traceback.print_exc()
            continue

    with open(args.output_path, "a") as outf:
        outf.write("\n]")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--retriever_path", required=True)
    p.add_argument("--clusters_metadata_path", required=True)
    p.add_argument("--kb_wikipedia_path", required=True)
    p.add_argument("--index_path",         required=True, help="dir con knn.index")
    p.add_argument("--index_json_path",    required=True, help="dir con knn.json")
    p.add_argument("--top_k",              type=int, default=10, help="numero di doc da rerankare")

    main(p.parse_args())
