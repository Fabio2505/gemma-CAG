#retrieval con generazione image+text

import argparse
import json
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import faiss
import numpy as np
import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModel,
    CLIPImageProcessor,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
import traceback

def load_clip_and_index(args):
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
    return "\n\n".join(texts)

def build_chat_prompt(context, question, image):
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content}
    ]
    return messages



def generate_answer(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.2,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()

def main(args):
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=True
    ).to("cuda:0").eval()

    clip_model, clip_processor, index, index_map, wiki = load_clip_and_index(args)
    with open(args.input_path) as f:
        dataset = json.load(f)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    predictions = []
    written_any = False
    with open(args.output_path, "w") as outf:
        outf.write("[\n")
    
    print("Loaded dataset size:", len(dataset))
    for i, sample in enumerate(tqdm(dataset[:10], desc="Generating"), 1):
        q = sample["question"]
        img_path = sample.get("related_images", None)
        ref = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]

        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Immagine non trovata: {img_path}")
            with Image.open(img_path) as im:
                im.verify()
            feats = extract_features(img_path, clip_model, clip_processor)
            ctx = retrieve_topk_pages(feats, index, index_map, wiki, k=args.top_k)
            #image = Image.open(img_path).convert("RGB"),messages = build_chat_prompt(ctx, q, image)
            messages = build_chat_prompt(ctx, q, img_path)


            print(f"[{i}] Q: {q}")#\nContext token len: {len(ctx)}") #\nImage: {img_path}") context è caratteri non token

            answer = generate_answer(model, processor, messages)

            print(f"[{i}] Risposta: {answer}")

            predictions.append({
                "question": q,
                "reference": ref,
                "answers": answer,
                "question_type": sample.get("question_type", "unknown"),
            })

        except FileNotFoundError as e:
            print(f"[IMG] FileNotFoundError su {img_path}: {e}")
            traceback.print_exc()
            continue
        except UnidentifiedImageError as e:
            print(f"[IMG] UnidentifiedImageError su {img_path}: {e}")
            traceback.print_exc()
            continue
        except OSError as e:
            print(f"[IMG] OSError su {img_path}: {e}")
            traceback.print_exc()
            continue
        except Exception as e:
            print(f"[{i}] Exception su domanda: {q}")
            print("Tipo:", type(e))
            print("Messaggio:", str(e)[:300])
            traceback.print_exc()
            continue

         

        if i % 100 == 0 or i == len(dataset) or i == 1:
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
            outf.write("]")

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
