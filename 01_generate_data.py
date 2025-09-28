import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from colorama import Fore, Style
from pydantic import BaseModel
from typing import List
import os
import json
import time

import sys
sys.path.append("..")
from scripts.llm_call import llm_call


## PyDantic schema for the extracted data
class QAPair(BaseModel):
    question: str
    answer: str

class generatedData(BaseModel):
    generated: List[QAPair]


#Prompt template for generating Q&A pairs
def get_prompt_template(data: str, num_records: int = 5) -> str:
    prompt = f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
    the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model for Bloomberg Terminal usage. Based on your own expertise, you are also encourage to add any relevant information and additional context based on your own expertise to enhance the quality of the Q&A pairs but make sure it's relevant.

    For each of the {num_records} entries, generate one or two well-structured questions that reflect different aspects of the information in the chunk. 
    Ensure a mix of longer and shorter questions. Each Q&A pair should be concise yet informative, capturing key insights from the data.

    Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:

        "question": "Your question here...",
        "answer": "Your answer here..."

    Focus on creating clear, relevant, and varied questions that encourage the model to learn from diverse perspectives. Avoid any sensitive or biased 
    content, ensuring answers are accurate and neutral.
    
    Data Chunk:
    ---
    {data}
    ---
    """
    return prompt

## To stick different JSON files together.
def save_dataset(dataset, output_path):
    # Make sure the parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If file exists, load existing data
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Merge datasets (key-based merge)
    existing_data.update(dataset)

    # Write back clean JSON
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)

def main():
    """Main funtion to run the data generation script"""

    start_time = time.time()  # ⏳ Start timer

    ## Print current directory
    print(Fore.CYAN + Style.BRIGHT + f"Current working directory: {os.getcwd()}" + Style.RESET_ALL)
    source_path = "data/raw_pdf/"
    output_path = "data/synthetic_QA_pairs/synthetic_output.json"
    generator_model = "ollama/llama3.1:8B"

    print(Fore.GREEN + Style.BRIGHT + "Starting synthetic data generation..." + Style.RESET_ALL)

    if not os.path.exists(source_path):
        print(Fore.RED + Style.BRIGHT + f"Error: Source PDF path {source_path} doesn't exist " + Style.RESET_ALL)
        return

    # Check if pdf files are present in the source path
    pdf_files = [f for f in os.listdir(source_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(Fore.RED + Style.BRIGHT + f"Error: No PDF files found in {source_path}" + Style.RESET_ALL)
        return
    
    # Open PDF files present in the source path one by one
    for pdf_file in pdf_files:
        print(Fore.GREEN + Style.BRIGHT + f"Step 1: Loading and chunking {pdf_file}..." + Style.RESET_ALL)
        pdf_path = os.path.join(source_path, pdf_file)

        full_text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        chunks = text_splitter.split_text(full_text)
        print(Fore.GREEN + Style.BRIGHT + f"Successfully loaded and chunked {pdf_file} into {len(chunks)} chunks." + Style.RESET_ALL)

        ## Generating Q&A pairs for each chunk
        print(Fore.GREEN + Style.BRIGHT + f"Step 2: Generating Q&A pairs for {pdf_file}..." + Style.RESET_ALL)
        
        dataset = {}
        for i, chunk in enumerate(chunks):
            print(Fore.MAGENTA + f"\n Processing chunk {i+1}/{len(chunks)}..." + Style.RESET_ALL)
            prompt = get_prompt_template(chunk, num_records=5)

            response = llm_call(
                model=generator_model,
                prompt=prompt,
                json_schema=json.dumps(generatedData.model_json_schema())
            )

            if response and "generated" in response:
                key = f"{pdf_file}_chunk_{i+1}"
                dataset[key] = {"generated": response["generated"], "context": chunk}
                print(Fore.GREEN + f"Successfully generated Q&A for chunk {i+1}." + Style.RESET_ALL)
            else:
                print(Fore.RED + f"Failed to generate Q&A for chunk {i+1}." + Style.RESET_ALL)

        # Save the dataset to a JSON file
        if dataset:
            save_dataset(dataset, output_path)
            print(Fore.GREEN + Style.BRIGHT +
                f"Step 3: Synthetic data for {pdf_file} generation completed and saved to {output_path}." +
                Style.RESET_ALL)
        else:
            print(Fore.RED + Style.BRIGHT + f"No data generated for {pdf_file}, nothing to save." + Style.RESET_ALL)
    
    print(Fore.GREEN + Style.BRIGHT + "Synthetic data generation process completed for all files." + Style.RESET_ALL)

    # ⏳ Print elapsed time
    end_time = time.time()
    total_time = end_time - start_time
    print(Fore.CYAN + Style.BRIGHT + f"\nTotal execution time: {total_time:.2f} seconds" + Style.RESET_ALL)

    ## Write the tiime taken to a file named overview.txt. Create the file if it doesn't exist.
    with open("overview.txt", "a") as f:
        f.write(f"Synthetic data generation time on AMD Turin SoC: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()