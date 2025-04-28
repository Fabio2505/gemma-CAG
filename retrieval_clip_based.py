import argparse
import torch
import json
import os
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, Gemma3ForCausalLM, AutoModel, CLIPImageProcessor, pipeline, CLIPModel, AutoProcessor
import faiss
import numpy as np


from transformers import AutoModel, CLIPImageProcessor

def load_clip_and_index(args):
   

    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda").eval()

    
    clip_processor = CLIPImageProcessor.from_pretrained(
        args.retriever_path
    )

    index = faiss.read_index(os.path.join(args.index_path, 'knn.index'))
 
    with open(os.path.join(args.index_json_path, 'knn.json'), 'r') as f:
        index_map = json.load(f)

    with open(args.kb_wikipedia_path, 'r') as f:
        wiki = json.load(f)

    return clip_model, clip_processor, index, index_map, wiki






def extract_features(image_path, clip_model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.float16, device="cuda")

    with torch.no_grad():
        features = clip_model.encode_image(pixel_values=pixel_values)
        features = features / features.norm(p=2, dim=-1, keepdim=True)

    
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise ValueError(f"Feature extraction failed: NaN or Inf in features for image {image_path}")

    return features.cpu().numpy().astype(np.float32)



def retrieve_topk_pages(features, index, index_map, wiki, k):
    D, I = index.search(features, k)
    
    if I.shape[1] < k:
        print(f"Warning: less than {k} pages retrieved, got {I.shape[1]}")
    
    urls = [index_map[i][0] for i in I[0]]
    texts = ["\n".join(wiki[url]["section_texts"]) for url in urls]

    
    if any(len(text.strip()) == 0 for text in texts):
        raise ValueError("Retrieved empty context text!")

    return texts

def build_prompt(context, question):

    if len(context) > 6000:
        context = context[:6000]

    
    prompt = (
    f"Contesto:\n{context}\n\n"
    f"Domanda:\n{question}\n\n"
    "Rispondi in modo breve e preciso basandoti SOLO sulle informazioni date. "
    "Se non sai rispondere, scrivi 'Non so'.\n"
    "Risposta:"
)
    
    if len(prompt.strip()) == 0:
        raise ValueError("Prompt vuoto generato!")

    return prompt

    




def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=10,
            do_sample=False,
            temperature=0.3,
            top_p=0.9,
            remove_invalid_values=True
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)




def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_model_path)
    model = Gemma3ForCausalLM.from_pretrained(
        args.gemma_model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    clip_model, clip_processor, index, index_map, wiki = load_clip_and_index(args)

    print("modelli ok")

    with open(args.input_path, 'r') as f:
        dataset = json.load(f)

    print("dataset caricato")

    predictions = []
    for sample in tqdm(dataset[:4]):
        question = sample['question']
        image_path = sample['related_images']
        reference = sample['answer']

        try:
            features = extract_features(image_path, clip_model, clip_processor)
            contexts = retrieve_topk_pages(features, index, index_map, wiki, k=args.top_k)
            full_context = "\n\n".join(contexts)
            prompt = build_prompt(full_context, question)
            

            print("\n=== Prompt ===\n", prompt) 
            print("Length of context:", len(full_context))

            answer = generate_answer(model, tokenizer, prompt)



            predictions.append({
                "question": question,
                "reference": reference,
                "answers": answer,
                "question_type": sample.get("question_type", "unknown")
            })
        except Exception as e:
            print(f"Errore nel sample: {question[:50]}... -> {e}")

    with open(args.output_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Percorso al file JSON di input.")
    parser.add_argument("--output_path", type=str, required=True, help="Percorso al file JSON di output.")
    parser.add_argument("--gemma_model_path", type=str, required=True, help="Percorso alla directory del modello Gemma.")
    parser.add_argument("--retriever_path", type=str, required=True, help="Percorso alla directory contenente i file del modello CLIP (model/, image_processor/).")
    parser.add_argument("--index_path", type=str, required=True, help="Percorso alla directory contenente l'indice FAISS (knn.index).")
    parser.add_argument("--index_json_path", type=str, required=True, help="Percorso alla directory contenente il file JSON di mapping dell'indice (knn.json).")
    parser.add_argument("--kb_wikipedia_path", type=str, required=True, help="Percorso al file JSON contenente la knowledge base di Wikipedia.")
    parser.add_argument("--top_k", type=int, default=3, help="Numero di pagine da recuperare.")
    args = parser.parse_args()
    main(args)



