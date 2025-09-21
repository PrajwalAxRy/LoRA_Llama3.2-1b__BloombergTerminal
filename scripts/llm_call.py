import json
from litellm import completion
from colorama import Fore

def llm_call(model: str, prompt: str, json_schema: str = None) -> dict:
    """
    Calls a language model to generate a response based on the provided prompt.
    If a JSON schema is provided, the response will be formatted according to that schema.

    Args:
        model (str): The name of the language model to use.
        prompt (str): The input prompt for the language model.
        json_schema (str, optional): A JSON schema to format the output. Defaults to None.
    """

    #First print what model we are using
    print(Fore.CYAN + f"--- Calling model: {model} ---" + Fore.RESET) # Output: --- Calling model: ollama/llama3.2:1B ---

    try:
        messages = [{"role": "user", "content": prompt}]

        api_params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"max_new_tokens": 2048, "temperature": 0.2},
        }

        if json_schema:
            api_params["format"] = json.loads(json_schema)
        
        stream = completion(**api_params)  # ** unpacks the dictionary into keyword arguments
        ## completion returns a generator when stream=True. So, We use a for loop to iterate over the generator

        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(Fore.LIGHTCYAN_EX + delta + Fore.RESET, end='', flush=True)  # Print the new content as it arrives. # flush=True ensures immediate output
                response += delta

        print("\n" + Fore.CYAN + "--- End of response ---" + Fore.RESET) 

        #convert response to python dict
        if json_schema:
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                print(Fore.RED + f"Failed to parse response as JSON: {e}" + Fore.RESET)
                return None
            
        return response
                 
    except Exception as e:
        print(Fore.RED + f"An error occurred during LLM call: {e}" + Fore.RESET)
        return None