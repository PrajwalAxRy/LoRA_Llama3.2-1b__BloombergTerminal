import os
import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig # SFTConfig is used for training arguments
from peft import LoraConfig, prepare_model_for_kbit_training
from colorama import Fore, Style

### Newer peft lib is used, ensuring print_trainable_parameters is correctly defined
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for p in model.parameters():
        num_params = p.numel()
        all_params += num_params
        if p.requires_grad:
            trainable_params += num_params
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")


# --- Centralized Configuration ---
TRAINING_CONFIG = {
    "base_model": "meta-llama/Llama-3.2-1B-instruct",
    "dataset_path": "data/synthetic_QA_pairs/03_synthetic_output_quality_checked.json",
    "output_dir": "checkpoints/bloomberg-llama-3.2-1b",
    "final_model_dir": "final_model",
    "epochs": 10,
    "lora_r": 16, ## LoRA rank
    "lora_alpha": 32, ## To make the scaling factor 2.0
    "lora_dropout": 0.05, ## Slightly higher dropout for regularization
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
    start_time = time.time()

    # Set default dtype for better GPU performance with 4-bit quantization
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.bfloat16)

    print(Fore.GREEN + Style.BRIGHT + "--- Starting Model Fine-Tuning ---" + Style.RESET_ALL)

    # --- 1. Load Hugging Face Token ---
    # NOTE: The Hugging Face token is typically loaded from the environment for models like Llama.
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        # Proceeding without token if it's not a private model, but raising a warning
        print(Fore.RED + "Warning: HUGGING_FACE_TOKEN environment variable not set. This may cause issues if the model is private or requires authentication." + Style.RESET_ALL)

    # --- 2. Load and Prepare the Dataset and Tokenizer ---
    print(Fore.YELLOW + f"Loading and preparing dataset from '{TRAINING_CONFIG['dataset_path']}'..." + Style.RESET_ALL)
    
    # Load the tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        TRAINING_CONFIG['base_model'],
        token=hf_token,
        trust_remote_code=True
    )
    # Llama 3 models do not have a default padding token, so we set it to the end-of-sequence token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and map the dataset
    try:
        dataset = load_dataset("json", data_files=TRAINING_CONFIG['dataset_path'], split='train')
    except FileNotFoundError:
        print(Fore.RED + f"Error: Dataset file not found at '{TRAINING_CONFIG['dataset_path']}'." + Style.RESET_ALL)
        return
        
    train_dataset = dataset.map(lambda batch: format_chat_template(batch, tokenizer), batched=True) ## Format the dataset
    print(Fore.GREEN + "Dataset prepared successfully." + Style.RESET_ALL)
    print(Fore.CYAN + "Example of formatted data:\n" + train_dataset[0]['text'] + Style.RESET_ALL)

    # --- 3. Configure Quantization ---
    print(Fore.YELLOW + "\n Configuring 4-bit quantization..." + Style.RESET_ALL)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(Fore.GREEN + "Quantization configured." + Style.RESET_ALL)

    # --- 4. Load the Base Model ---
    print(Fore.YELLOW + f"\nLoading base model '{TRAINING_CONFIG['base_model']}'..." + Style.RESET_ALL)
    model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG['base_model'],
        device_map="auto", # Automatically uses the GPU
        quantization_config=quant_config,
        token=hf_token,
        cache_dir="./model_cache",
    )
    print(Fore.GREEN + "Base model loaded successfully." + Style.RESET_ALL)

    # --- 5. Prepare Model for LoRA Training ---
    model.gradient_checkpointing_enable()  # Saves memory
    model = prepare_model_for_kbit_training(model)

    print(Fore.MAGENTA + "Trainable parameters after prepare_model_for_kbit_training:" + Style.RESET_ALL)
    print_trainable_parameters(model) 

    # --- 6. Configure LoRA ---
    print(Fore.YELLOW + "\n Configuring LoRA..." + Style.RESET_ALL)
    peft_config = LoraConfig(
        r=TRAINING_CONFIG['lora_r'],
        lora_alpha=TRAINING_CONFIG['lora_alpha'],
        lora_dropout=TRAINING_CONFIG['lora_dropout'],
        target_modules="all-linear", # Fine-tune all linear layers for best results
        task_type="CAUSAL_LM", # Causal Language Modeling task
    )

    print(Fore.CYAN + f"LoRA Rank (r): {TRAINING_CONFIG['lora_r']}, Alpha: {TRAINING_CONFIG['lora_alpha']}" + Style.RESET_ALL)
    print(Fore.GREEN + "LoRA configured." + Style.RESET_ALL)
    

    print(Fore.YELLOW + "\nStarting the training process..." + Style.RESET_ALL)
    
    # --- 7. Initialize and Run SFTTrainer (FIXED) ---
    # Based on current TRL documentation, these parameters go in SFTConfig
    training_args = SFTConfig(
        output_dir=TRAINING_CONFIG['output_dir'],
        num_train_epochs=TRAINING_CONFIG['epochs'],
        per_device_train_batch_size=12, # Adjust based on your GPU memory
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        # These parameters belong in SFTConfig in current TRL version
        dataset_text_field="text",
        max_length=512,  # Note: it's max_length, not max_seq_length in SFTConfig
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
    )
    
  
    # Start training
    trainer.train()
    
    print(Fore.GREEN + Style.BRIGHT + "\nTraining complete!" + Style.RESET_ALL)

    # --- 8. Save the Final Model ---
    print(Fore.YELLOW + f"\nSaving the final model to '{TRAINING_CONFIG['final_model_dir']}'..." + Style.RESET_ALL)
    trainer.save_model(TRAINING_CONFIG['final_model_dir'])
    
    print(Fore.GREEN + Style.BRIGHT + f"\nModel saved successfully. You can now use the adapters in '{TRAINING_CONFIG['final_model_dir']}' for inference." + Style.RESET_ALL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = f"{elapsed_time/60:.2f} minutes"
    print(Fore.BLUE + Style.BRIGHT + f"\nTotal time taken: {elapsed_str}" + Style.RESET_ALL)

    # Assuming 'overview.txt' exists or should be created
    with open("overview.txt", "a") as f:
        f.write(f"Time taken for training: {elapsed_str}\n")


if __name__ == "__main__":
    main()