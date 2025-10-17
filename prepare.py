# preprocess_for_training.py
import json, re, hashlib, argparse
from pathlib import Path
from tqdm import tqdm

INFILE = ".//9tuc6e.jsonl"
OUTFILE = "./data_prepared.jsonl"
OUT_SAMPLE = "./data_prepared_100k.jsonl"

def extract_input_from_prompt(prompt_text):
    # Find "### Input:" and "### Response:" sections; robust to spacing.
    # Return the text between them (trimmed).
    m = re.search(r"###\s*Input\s*:\s*(.*?)\s*###\s*Response\s*:", prompt_text, flags=re.S|re.I)
    if m:
        return m.group(1).strip()
    # fallback: look for "### Input:" then end
    m2 = re.search(r"###\s*Input\s*:\s*(.*)", prompt_text, flags=re.S|re.I)
    if m2:
        return m2.group(1).strip()
    return ""

def normalize_prompt(prompt_text):
    # Keep instruction and Response marker, but strip the input body.
    # We'll create a predictable prompt template the model sees before generation.
    # Extract Instruction if present:
    ins = ""
    m = re.search(r"(###\s*Instruction\s*:\s*(.*?))(?=###\s*Input\s*:)", prompt_text, flags=re.S|re.I)
    if m:
        ins = m.group(1).strip()
    else:
        # fallback: keep everything up to "### Input:"
        parts = re.split(r"###\s*Input\s*:", prompt_text, maxsplit=1, flags=re.I)
        ins = parts[0].strip() if parts else prompt_text.strip()
    # Ensure there is a Response marker at end
    normalized = f"{ins}\n\n### Response:"
    return normalized

def main():
    infile = Path(INFILE)
    out = Path(OUTFILE)
    out_sample = Path(OUT_SAMPLE)

    seen = set()
    total = 0
    written = 0
    kept = []

    with infile.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Reading"):
            total += 1
            line=line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except:
                continue
            prompt = j.get("prompt","")
            completion = j.get("completion", "")
            # If completion is empty, try to extract from prompt's Input section
            if (not str(completion).strip()):
                extracted = extract_input_from_prompt(prompt)
                response_text = extracted
            else:
                response_text = str(completion).strip()

            if not response_text or len(response_text.strip()) < 2:
                # skip empty / trivial responses
                continue

            # normalize prompt template
            new_prompt = normalize_prompt(prompt)

            # dedupe by the pair (prompt, response)
            key = hashlib.sha256((new_prompt + "\n|||\n" + response_text).encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)

            out_obj = {
                "prompt": new_prompt,
                "response": response_text
            }
            kept.append(out_obj)

    # Save full cleaned file
    with out.open("w", encoding="utf-8") as fout:
        for obj in kept:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Save a 100k sample for experiments (or full if smaller)
    sample_k = min(100000, len(kept))
    import random
    random.seed(42)
    sampled = random.sample(kept, sample_k)
    with out_sample.open("w", encoding="utf-8") as fout:
        for obj in sampled:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Total lines read: {total}")
    print(f"Kept (unique, non-empty) examples: {len(kept)}")
    print(f"Wrote: {out} and sample {out_sample} ({sample_k} examples)")

if __name__ == "__main__":
    main()
