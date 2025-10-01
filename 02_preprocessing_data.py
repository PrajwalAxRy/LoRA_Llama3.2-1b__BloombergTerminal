import json
from colorama import Fore, Style

def main():
    """ 
        Reads the synthetic data from 'synthetic_data.json', processes it to extract Q&A pairs.
    """

    INPUT_FILE = "data/synthetic_QA_pairs/01_synthetic_output.json"
    OUTPUT_FILE = "data/synthetic_QA_pairs/02_synthetic_output_processed.json"

    print(Fore.GREEN + Style.BRIGHT + "--- Starting Phase 2, Step 1: Preprocessing ---" + Style.RESET_ALL)

    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(Fore.RED + Style.BRIGHT + f"Error: The file {INPUT_FILE} was not found." + Style.RESET_ALL)
        return
    
    """
    Current JSON format
    {
    "0": {
        "generated": [],
        "context": "GETTING  \nSTARTED ON  \nBLOOMBERG\nLAUNCHPAD\nStart your day with the most powerful and \nflexible desktop workspace available \nfor financial professionals.\nUSER GUIDE\n\u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0 \u00b0\nA Bloomberg Professional Service Offering\nCONTENTS\n04 ACCESSING LAUNCHPAD FOR THE FIRST TIME\n04 COMMON BLP COMMANDS\n04 SETTING UP LAUNCHPAD VIEWS\n05 MANAGING VIEWS\n05 OPENING AN EXISTING VIEW OR SAMPLE VIEW\n06 OPENING LAUNCH COMPONENTS\n06 OPENING A NEW MONITOR FROM THE TOOLBAR\n07 ENTERING SECURITIES INTO A MONITOR\n07 EDITING MONITOR COLUMN DATA\n08 ADDING FUNCTION SHORTCUTS\n09 SCREEN ADJUSTMENT\n10 NEWS PANELS\n11 CHARTS\n11 ADDING TECHNICAL STUDIES\n12 CHART GRID\n12 ADDING A CUSTOM FUNCTION\n13 LINKING COMPONENTS\n14 QUICKLY ADD A FUNCTION TO LAUNCHPAD\n14 RESTORING A MONITOR"
    },
    "1": {
        "generated": [
        {
            "question": "What is Bloomberg Launchpad used for?",
            "answer": "Bloomberg Launchpad is designed to help users combine multiple functions and monitors on pages and in views, organizing and consolidating their desktop to fit their personal workflow."
        },
        {
            "question": "How does Launchpad's monitoring feature work?",
            "answer": "Launchpad's monitoring features allow users to "
        }
        ...
    ...
    }
    """
    
    instructions = []
    for key, chunk in data.items():
        if 'generated' in chunk:
            for pair in chunk['generated']:
                if pair.get('question') and pair.get('answer'):
                    instructions.append({
                        "question": pair['question'],
                        "answer": pair['answer']
                    })  

    print(Fore.CYAN + Style.BRIGHT + f"Total Q&A pairs extracted: {len(instructions)}" + Style.RESET_ALL)

    ## Write the processed data to a new JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(instructions, f, indent=4)
    
    print(Fore.GREEN + Style.BRIGHT + f"Processed data saved to {OUTPUT_FILE}" + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + "--- Preprocessing Complete ---" + Style.RESET_ALL)
                        
if __name__ == "__main__":
    main()