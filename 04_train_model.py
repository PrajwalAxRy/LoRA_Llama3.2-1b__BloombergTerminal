import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from colorama import Fore, Style

# --- Centralized Configuration ---
TRAINING_CONFIG = {
    "base_model": "meta-llama/Llama-3.2-1B-instruct",
    "dataset_path": "data/bloomberg_instruction_quality.json",
    "output_dir": "checkpoints/bloomberg-llama-3.2-1b",
    "final_model_dir": "final_model",
    "epochs": 20, # A lower number of epochs is often sufficient for LoRA
    "lora_r": 128,
    "lora_alpha": 256,
    "lora_dropout": 0.05,
}

def format_chat_template(batch, tokenizer):
    """
    Converts question-answer pairs into proper chat format for instruction following
    """
    system_prompt = "You are a helpful assistant specializing in the Bloomberg Terminal. Provide clear, accurate answers to questions about its functions and usage."
    
    questions = batch["question"]
    answers = batch["answer"]
    
    texts = []
    for question, answer in zip(questions, answers):
        # Create the conversation structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        # Check if tokenizer has a chat template
        if tokenizer.chat_template is not None:
            # Use the tokenizer's built-in chat template
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Manually format using Llama-3 style template
            formatted_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            formatted_text += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
            formatted_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        
        texts.append(formatted_text)
        
    return {"text": texts}

def main():
    """Main function to run the model training pipeline."""
    
    print(Fore.GREEN + Style.BRIGHT + "--- Starting Phase 3: Model Fine-Tuning ---" + Style.RESET_ALL)

    # --- 1. Load Hugging Face Token ---
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        print(Fore.RED + "Error: Hugging Face token not found. Please set the HUGGING_FACE_TOKEN environment variable." + Style.RESET_ALL)
        return

    # --- 2. Load and Prepare the Dataset ---
    print(Fore.YELLOW + f"Step 1: Loading and preparing dataset from '{TRAINING_CONFIG['dataset_path']}'..." + Style.RESET_ALL)
    dataset = load_dataset("json", data_files=TRAINING_CONFIG['dataset_path'], split='train')
    
    tokenizer = AutoTokenizer.from_pretrained(
        TRAINING_CONFIG['base_model'],
        token=hf_token,
        trust_remote_code=True
    )
    # Llama 3 models do not have a default padding token, so we set it to the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = dataset.map(lambda batch: format_chat_template(batch, tokenizer), batched=True)
    print(Fore.GREEN + "Dataset prepared successfully." + Style.RESET_ALL)
    print(Fore.CYAN + "Example of formatted data:\n" + train_dataset[0]['text'] + Style.RESET_ALL)

    # --- 3. Configure Quantization ---
    print(Fore.YELLOW + "\nStep 2: Configuring 4-bit quantization..." + Style.RESET_ALL)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(Fore.GREEN + "Quantization configured." + Style.RESET_ALL)

    # --- 4. Load the Base Model ---
    print(Fore.YELLOW + f"\nStep 3: Loading base model '{TRAINING_CONFIG['base_model']}'..." + Style.RESET_ALL)
    model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG['base_model'],
        device_map="auto", # Automatically uses the GPU
        quantization_config=quant_config,
        token=hf_token,
        cache_dir="./model_cache",
    )
    print(Fore.GREEN + "Base model loaded successfully." + Style.RESET_ALL)

    # --- 5. Prepare Model for LoRA Training ---
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- 6. Configure LoRA ---
    print(Fore.YELLOW + "\nStep 4: Configuring LoRA..." + Style.RESET_ALL)
    peft_config = LoraConfig(
        r=TRAINING_CONFIG['lora_r'],
        lora_alpha=TRAINING_CONFIG['lora_alpha'],
        lora_dropout=TRAINING_CONFIG['lora_dropout'],
        target_modules="all-linear", # Apply LoRA to all linear layers
        task_type="CAUSAL_LM",
    )
    print(Fore.GREEN + "LoRA configured." + Style.RESET_ALL)
    

    print(Fore.YELLOW + "\nStep 5: Starting the training process..." + Style.RESET_ALL)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024, # Set a reasonable sequence length
        args=SFTConfig(
            output_dir=TRAINING_CONFIG['output_dir'],
            num_train_epochs=TRAINING_CONFIG['epochs'],
            per_device_train_batch_size=4, # Adjust based on your GPU memory
            gradient_accumulation_steps=2,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
        ),
    )
    
    # Start training
    trainer.train()
    
    print(Fore.GREEN + Style.BRIGHT + "\nTraining complete!" + Style.RESET_ALL)

    # --- 8. Save the Final Model ---
    print(Fore.YELLOW + f"\nStep 6: Saving the final model to '{TRAINING_CONFIG['final_model_dir']}'..." + Style.RESET_ALL)
    trainer.save_model(TRAINING_CONFIG['final_model_dir'])
    
    print(Fore.GREEN + Style.BRIGHT + f"\nModel saved successfully. You can now use the adapters in '{TRAINING_CONFIG['final_model_dir']}' for inference." + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + "\n--- Phase 3 Complete ---" + Style.RESET_ALL)


if __name__ == "__main__":
    main()