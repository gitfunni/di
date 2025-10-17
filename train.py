import os
import time
import math
import sys
import torch
from datasets import load_dataset
from typing import Dict, Any

# Prevent Transformers from importing torchvision (breaks on CPU-only/unsupported builds)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Try Unsloth first; if it refuses to import without a GPU, we fall back to HF Transformers+PEFT on CPU.
try:
    from unsloth import FastLanguageModel  # type: ignore
    _UNSLOTH_OK = True
except Exception as _e:
    print("Unsloth unavailable or requires a GPU; falling back to Transformers+PEFT on CPU.\nReason:", _e)
    _UNSLOTH_OK = False

# --- Config ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
data_file = "data_prepared_100k.jsonl"
output_dir = "tinyllama-ft"
max_seq_length = 256
batch_size = 2
grad_accum = 2
epochs = 1
pilot_steps = 100
RUN_FULL_TRAIN = False
# Force CPU visibility by default, so we don’t inadvertently hit a partial GPU setup
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# --------------

# CPU threading
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "4")
try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
except Exception:
    pass

print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": data_file}, split="train")

def _ensure_fields(example: Dict[str, Any]) -> Dict[str, str]:
    # Simple guard if keys differ
    prompt = example.get("prompt") or example.get("input") or ""
    response = example.get("response") or example.get("output") or ""
    return {"prompt": str(prompt), "response": str(response)}

dataset = dataset.map(_ensure_fields)

if _UNSLOTH_OK:
    # Unsloth path (GPU only). If you actually have a supported GPU, remove CUDA_VISIBLE_DEVICES restriction above.
    try:
        # If running on CPU-only env, Unsloth will be too slow or unsupported.
        # But Unsloth does offer CPU patches for inference; training still expects an accelerator.
        FastLanguageModel.patch_cpu_backend("torch.compile")
    except Exception:
        pass

    print("Loading TinyLlama with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
        load_in_4bit=False,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=4,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
    )

    eos = tokenizer.eos_token or "</s>"

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        prompts = batch["prompt"]
        responses = batch["response"]
        input_ids_list = []
        attn_masks_list = []
        labels_list = []
        for p, r in zip(prompts, responses):
            # Tokenize prompt and combined sequence
            p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            full = p + r + eos
            enc = tokenizer(
                full,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
            )
            inp = enc["input_ids"]
            msk = enc["attention_mask"]
            # Mask prompt tokens in labels
            lab = inp.copy()
            prompt_len = min(len(p_ids), max_seq_length)
            for i in range(prompt_len):
                lab[i] = -100
            input_ids_list.append(inp)
            attn_masks_list.append(msk)
            labels_list.append(lab)
        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_masks_list,
            "labels": labels_list,
        }

    dataset_tok = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset size: {len(dataset_tok)} examples")

    trainer = FastLanguageModel.get_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset_tok,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-5,
        num_train_epochs=epochs,
        max_seq_length=max_seq_length,
    )

    trainer.args.max_steps = pilot_steps
    print(f"Running pilot ({pilot_steps} steps) with Unsloth...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    avg_s = elapsed / max(1, pilot_steps)
    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, len(dataset_tok) // max(1, eff_batch))
    eta_h = steps_per_epoch * avg_s / 3600
    print(f"Avg {avg_s:.3f}s/step → ETA ≈ {eta_h:.2f} h/epoch")

    if RUN_FULL_TRAIN:
        print("Running full training...")
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        try:
            FastLanguageModel.to_ollama(output_dir, output_dir + "-ollama")
        except Exception:
            pass
else:
    # Transformers+PEFT CPU fallback
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
    from peft import LoraConfig, get_peft_model

    print("Loading TinyLlama with Transformers on CPU (this will be slow)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to("cpu")

    # LoRA config similar to the Unsloth path
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    eos = tokenizer.eos_token or "</s>"

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        prompts = batch["prompt"]
        responses = batch["response"]
        input_ids_list = []
        attn_masks_list = []
        labels_list = []
        for p, r in zip(prompts, responses):
            p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            full = p + r + eos
            enc = tokenizer(
                full,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
            )
            inp = enc["input_ids"]
            msk = enc["attention_mask"]
            lab = inp.copy()
            prompt_len = min(len(p_ids), max_seq_length)
            for i in range(prompt_len):
                lab[i] = -100
            input_ids_list.append(inp)
            attn_masks_list.append(msk)
            labels_list.append(lab)
        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_masks_list,
            "labels": labels_list,
        }

    dataset_tok = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset size: {len(dataset_tok)} examples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=max(1, batch_size // 2),  # be conservative on CPU
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-5,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        ddp_find_unused_parameters=False,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        max_steps=pilot_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tok,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    print(f"Running pilot ({pilot_steps} steps) on CPU...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    avg_s = elapsed / max(1, pilot_steps)
    eff_batch = max(1, (max(1, batch_size // 2)) * grad_accum)
    steps_per_epoch = max(1, len(dataset_tok) // eff_batch)
    eta_h = steps_per_epoch * avg_s / 3600
    print(f"Avg {avg_s:.3f}s/step → ETA ≈ {eta_h:.2f} h/epoch")

    if RUN_FULL_TRAIN:
        print("Running full training on CPU (expect very slow speed)...")
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
