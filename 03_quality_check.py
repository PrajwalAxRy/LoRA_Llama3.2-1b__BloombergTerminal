import json
from pydantic import BaseModel
from colorama import Fore, Style
import os
import time

from tqdm import tqdm

import sys
sys.path.append("..")
from scripts.llm_call import llm_call


## Pydantic schema for the input data
class Score(BaseModel):
    score: int
    explanation: str

class Rank(BaseModel):
    accuracy: Score
    style: Score

def get_scoring_prompt(record: dict) -> str:
    """Creates the prompt for the LLM to score the Q&A pair."""
    return f"""You are a meticulous data quality analyst. Your task is to classify the following question-and-answer pair on a scale of 1-10 for both accuracy and style. Provide a brief explanation for your reasoning. The answer should be completely self-contained and based on general knowledge of the financial industry and the Bloomberg Terminal.

- **Accuracy Score**:
  - **0**: The question is not a real question or is nonsensical.
  - **1-3**: The answer is completely incorrect or does not adequately answer the question.
  - **4-6**: The answer is partially correct but has significant errors or omissions.
  - **7-9**: The answer is mostly accurate but could be more precise or detailed.
  - **10**: The answer is perfectly accurate, specific, and correct.

- **Style Score**:
  - **1-3**: The question or answer is harmful, unhelpful, dishonest, or contains gibberish.
  - **4-6**: The style is poor, unprofessional, or hard to understand.
  - **7-9**: The style is good, clear, and helpful.
  - **10**: The style is excellent, professional, and perfectly clear.

- **Record to Score**:
  - **Question**: {record['question']}
  - **Answer**: {record['answer']}

- **Return your response in this exact JSON format**:
{Rank.model_json_schema()}
"""

def main():
    """ Function to run data curtation """

    start_time = time.time()

    INPUT_FILE = "data/synthetic_QA_pairs/02_synthetic_output_processed.json"
    QUALITY_OUTPUT_FILE = "data/synthetic_QA_pairs/03_synthetic_output_quality_checked.json"
    FULL_RESULTS_FILE = "data/synthetic_QA_pairs/03_synthetic_output_assessment_results.json"
    SCORING_MODEL = "ollama/llama3.1:8B"

    ACCURACY_THRESHOLD = 7
    STYLE_THRESHOLD = 7

    print(Fore.CYAN + "Starting quality check process..." + Style.RESET_ALL)

    try:
        with open(INPUT_FILE, "r") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(Fore.RED + f"Input file not found: {INPUT_FILE}" + Style.RESET_ALL)
        return
    
    ## Add Test Mode
    TEST_MODE = False
    if TEST_MODE:
        records = records[:5]
        print(Fore.YELLOW + "Running in TEST MODE: Only processing first 5 records." + Style.RESET_ALL)
    
    high_quality_records = []
    full_quality_results = []

    for i, pair in enumerate(tqdm(records, desc="Evaluating Q&A Pairs", unit="pair")):
        print(Fore.YELLOW + f"Evaluating Pair {i+1}/{len(records)}" + Style.RESET_ALL)
        print(Fore.BLUE + f"Q: {pair['question']} \n A: {pair['answer']}" + Style.RESET_ALL)

        prompt = get_scoring_prompt(pair)
        result = llm_call(model=SCORING_MODEL, prompt=prompt, json_schema=json.dumps(Rank.model_json_schema()))

        if result and 'accuracy' in result and 'style' in result:
            accuracy_score = result['accuracy']['score']
            style_score = result['style']['score']

            print(Fore.CYAN + f"Accuracy Score: {accuracy_score}/10, Style Score: {style_score}/10" + Style.RESET_ALL)

            full_quality_results.append({**pair, 'quality': result})

            if accuracy_score >= ACCURACY_THRESHOLD and style_score >= STYLE_THRESHOLD:
                high_quality_records.append(pair)
                print(Fore.GREEN + "Record accepted." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Record rejected due to low quality." + Style.RESET_ALL)
    
    print(Fore.YELLOW + f"Saving {len(high_quality_records)} high-quality records to {QUALITY_OUTPUT_FILE}" + Style.RESET_ALL)
    with open(QUALITY_OUTPUT_FILE, "w") as f:
        json.dump(high_quality_records, f, indent=2)

    print(Fore.YELLOW + f"Saving full quality assessment results to {FULL_RESULTS_FILE}" + Style.RESET_ALL)
    with open(FULL_RESULTS_FILE, "w") as f:
        json.dump(full_quality_results, f, indent=2)

    print(Fore.CYAN + "Quality check process completed." + Style.RESET_ALL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = f"{elapsed_time:.2f} seconds"
    print(Fore.CYAN + f"Quality check process completed in {elapsed_str}." + Style.RESET_ALL)

    with open("overview.txt", "a") as f:
        f.write(f"\nData quality check process done in {elapsed_str}")

if __name__ == "__main__":
    main()

            
