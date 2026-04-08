"""
training/dpo.py — Stage 2: Direct Preference Optimisation (DPO).

Implements the paper's DPO objective (Eq. 3-9):

    L_DPO(π_θ; π_ref) = -E[(x, Iʷ, Iˡ) ~ D] [
        log σ( β · (log π_θ(Iʷ|x) - log π_θ(Iˡ|x)
                   - log π_ref(Iʷ|x) + log π_ref(Iˡ|x)) )
    ]

Uses TRL DPOTrainer with the SFT-aligned model as π_ref.

Training parameters from the paper:
  - β = 0.1
  - Learning rate: 1e-5
  - Epochs: 1
  - Global batch size: 32
  - Max prompt length: 1024, max total length: 2048
  - Warmup ratio: 0.03
  - BF16
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_dpo_trainer(cfg: dict):
    """
    Construct and return a TRL DPOTrainer.

    The SFT model checkpoint is used as both the initial model weights (π_θ)
    and the frozen reference model (π_ref).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import DPOTrainer

    from training.dataset import load_dataset

    # ── Load DPO dataset ───────────────────────────────────────────────────
    records = load_dataset(cfg["data"]["dpo_dataset_path"])
    hf_dataset = Dataset.from_dict({
        "prompt":   [r["prompt"]   for r in records],
        "chosen":   [r["chosen"]   for r in records],
        "rejected": [r["rejected"] for r in records],
    })

    # ── Quantisation ──────────────────────────────────────────────────────
    bnb_cfg = None
    if cfg["llm"].get("load_in_4bit", True):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # ── Tokeniser ─────────────────────────────────────────────────────────
    sft_model_path = cfg["llm"]["sft_output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Policy model (π_θ) — start from SFT checkpoint ───────────────────
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        quantization_config=bnb_cfg,
        device_map=cfg["llm"].get("device_map", "auto"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ── LoRA for DPO fine-tuning ───────────────────────────────────────────
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
    dpo_cfg = cfg["training"]["dpo"]
    output_dir = cfg["llm"]["dpo_output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        lr_scheduler_type=dpo_cfg["lr_scheduler_type"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        bf16=dpo_cfg.get("bf16", True),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        optim="paged_adamw_32bit",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,       # TRL uses the same LoRA model as ref when ref_model=None
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        peft_config=peft_config,
        beta=dpo_cfg["beta"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        max_length=dpo_cfg["max_length"],
        args=training_args,
    )
    return trainer, output_dir


def run_dpo(cfg: dict) -> None:
    """Run the DPO training stage."""
    print("=== Stage 2: Direct Preference Optimisation (DPO) ===")
    trainer, output_dir = _build_dpo_trainer(cfg)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"DPO model saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Stage 2: DPO preference optimisation")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_dpo(cfg)


if __name__ == "__main__":
    _main()
