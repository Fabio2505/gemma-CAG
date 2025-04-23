from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, default="gemma_predictions_formatted.json")
args = parser.parse_args()

model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

with open(args.input_path, "r") as f:
    original_dataset = json.load(f)

#  [
with open(args.output_path, "w") as f:
    f.write("[\n")

formatted_data = []
written_any = False  # Per gestire le virgole

#for i, sample in enumerate(tqdm(original_dataset[:1], desc="Test 1 sample"), 1):
for i, sample in enumerate(tqdm(original_dataset, desc="Generazione risposte"), 1):
    try:
        q = sample["question"]
        ref = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]
        q_type = sample.get("question_type", "templated")

        prompt = f"{q}\nRisposta:"
        output = pipe(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
        prediction = output.split("Risposta:")[-1].strip()

        formatted_data.append({
            "question": q,
            "reference": ref,
            "answers": prediction,
            "question_type": q_type
        })

    except Exception as e:
        print(f"[{i}/{len(original_dataset)}] ⚠️ Errore nel sample: {e}")

    if i == 1 or i % 100 == 0 or i == len(original_dataset):
        print(f"[{i}/{len(original_dataset)}] ✅ Salvataggio batch e pulizia RAM...")
        with open(args.output_path, "a") as f:
            for entry in formatted_data:
                if written_any:
                    f.write(",\n")
                json.dump(entry, f)
                written_any = True
        formatted_data = []

# ]
with open(args.output_path, "a") as f:
    f.write("\n]\n")
