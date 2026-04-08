"""
training/sft.py — Stage 1: Supervised Fine-Tuning (SFT) with LoRA.

Uses the Hugging Face TRL SFTTrainer together with PEFT LoRA.

Training parameters from the paper:
  - LoRA: r=16, alpha=32, dropout=0.05
  - Learning rate: 2e-4  (cosine scheduler)
  - Epochs: 3
  - Global batch size: 32 (per_device=4, gradient_accumulation=8)
  - Max sequence length: 2048
  - Warmup ratio: 0.03
  - Precision: BF16
  - Quantisation: 4-bit NF4 (bitsandbytes)

The SFT dataset format expected:
  [{"prompt": "<text>", "completion": "<text>"}, ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_sft_trainer(cfg: dict):
    """
    Construct and return a TRL SFTTrainer.

    This function is intentionally kept separate so it can be tested without
    triggering imports at module level (avoids requiring GPU at import time).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    from training.dataset import load_dataset

    # ── Load dataset ───────────────────────────────────────────────────────
    records = load_dataset(cfg["data"]["sft_dataset_path"])
    # Combine prompt + completion into a single "text" field
    texts = [r["prompt"] + "\n" + r["completion"] for r in records]
    hf_dataset = Dataset.from_dict({"text": texts})

    # ── Quantisation ──────────────────────────────────────────────────────
    bnb_cfg = None
    if cfg["llm"].get("load_in_4bit", True):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # ── Model + Tokeniser ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["llm"]["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["llm"]["base_model"],
        quantization_config=bnb_cfg,
        device_map=cfg["llm"].get("device_map", "auto"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if bnb_cfg is not None:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_cfg = cfg["llm"]["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training arguments ────────────────────────────────────────────────
    sft_cfg = cfg["training"]["sft"]
    output_dir = cfg["llm"]["sft_output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        bf16=sft_cfg.get("bf16", True),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        optim="paged_adamw_32bit",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=sft_cfg["max_seq_length"],
        args=training_args,
    )
    return trainer, output_dir


def run_sft(cfg: dict) -> None:
    """Run the SFT training stage."""
    print("=== Stage 1: Supervised Fine-Tuning ===")
    trainer, output_dir = _build_sft_trainer(cfg)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"SFT model saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Stage 1: SFT fine-tuning")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_sft(cfg)


if __name__ == "__main__":
    _main()
