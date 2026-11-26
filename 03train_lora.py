#!/usr/bin/env python3
"""
LoRA/QLoRA-Training für E-Mail-Schreibstil.

Trainiert einen LoRA-Adapter auf den JSONL-Trainingsdaten aus
lora_email_train.py. Verwendet standardmäßig QLoRA (4-Bit), um auch
auf bescheidener GPU-Hardware zu funktionieren.

Voraussetzungen:
    pip install torch transformers peft trl datasets bitsandbytes accelerate

Verwendung:
    python3 train_lora.py --model /pfad/oder/hf-id/modell
    python3 train_lora.py --model /pfad/zu/modell --no-4bit   # volle Präzision
    python3 train_lora.py --model /pfad/zu/modell --epochs 5 --batch-size 2
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer
except ImportError as e:
    print(f"⚠️  Fehlendes Paket: {e}")
    print("Bitte installieren: pip install torch transformers peft trl datasets bitsandbytes accelerate")
    sys.exit(1)


DEFAULT_DATA = Path.home() / "training/lora_emails/lora_email_training.jsonl"
DEFAULT_OUTPUT = Path.home() / "training/lora_emails/lora_model"


def format_example(example: dict) -> dict:
    """Formatiert instruction/input/output in ein Trainings-Textfeld."""
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get('input'):
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}


def detect_target_modules(model) -> list[str]:
    """Erkennt gängige Attention-/MLP-Modulnamen automatisch."""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.rsplit('.', 1)[-1]
            names.add(leaf)
    # Für Llama/Mistral/Qwen/Phi-Modelle übliche Module
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "qkv_proj", "out_proj", "fc1", "fc2", "dense",
    ]
    found = [p for p in preferred if p in names]
    if not found:
        # Fallback: alle Linearen Module (außer lm_head)
        found = sorted(n for n in names if n not in ("lm_head",))
    return found


def main():
    parser = argparse.ArgumentParser(description="LoRA-Training für E-Mail-Stil")
    parser.add_argument("--model", required=True, help="Basis-Modell (HF-ID oder lokaler Pfad)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL-Trainingsdatei")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ausgabeverzeichnis")
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl Epochen")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch-Größe pro Gerät")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max. Sequenzlänge")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA-Rang")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA-Alpha")
    parser.add_argument("--no-4bit", action="store_true", help="Ohne 4-Bit-Quantisierung trainieren")
    parser.add_argument("--target-modules", nargs="+", default=None,
                        help="LoRA-Zielmodule (überschreibt Auto-Erkennung)")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"❌ Trainingsdaten nicht gefunden: {args.data}")
        print("   Zuerst lora_email_train.py ausführen.")
        return

    print(f"📂 Trainingsdaten: {args.data}")
    print(f"🤖 Basis-Modell:   {args.model}")
    print(f"💾 Ausgabe:        {args.output}")
    print(f"🔧 4-Bit:          {'nein' if args.no_4bit else 'ja (QLoRA)'}")

    # --- Dataset laden ---
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    dataset = dataset.map(format_example)
    print(f"✅ {len(dataset)} Trainingsbeispiele geladen")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Modell laden ---
    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    # --- LoRA-Konfiguration ---
    target_modules = args.target_modules or detect_target_modules(model)
    print(f"🎯 LoRA-Zielmodule: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        fp16=not args.no_4bit,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to="none",
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        packing=False,
    )

    print("\n🚀 Starte Training...")
    trainer.train()

    # --- Adapter speichern ---
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\n✅ LoRA-Adapter gespeichert: {args.output}")
    print(f"   Dateien: adapter_model.safetensors, adapter_config.json")


if __name__ == "__main__":
    main()
