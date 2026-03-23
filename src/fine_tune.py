"""
Fine-tune a small model (e.g., llama3.2:1b) on the best responses from the leaderboard.
This is a minimal example using Hugging Face Transformers and PEFT.
"""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model

def load_best_responses(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Expecting list of {"question": ..., "best_response": ...}
    return data

def prepare_dataset(data, tokenizer):
    texts = [f"Question: {item['question']}\nAnswer: {item['best_response']}" for item in data]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    return Dataset.from_dict(tokenized)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSON file with best responses")
    parser.add_argument("--model", default="llama3.2:1b", help="Base model name (Ollama name) or HuggingFace ID")
    parser.add_argument("--output_dir", default="./output", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # For simplicity, we'll use a HuggingFace model. If the model name is an Ollama model,
    # you'd need to map it to a HuggingFace ID. Here we assume a HuggingFace ID like "meta-llama/Llama-3.2-1B".
    # Adjust as needed.
    model_name = "meta-llama/Llama-3.2-1B"  # Replace with actual HF ID if needed

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    data = load_best_responses(args.data)
    dataset = prepare_dataset(data, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()