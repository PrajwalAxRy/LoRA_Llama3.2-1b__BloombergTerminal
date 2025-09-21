import json
from llm_call import llm_call

if __name__ == "__main__":
    
    prompt = "Do gods of death like apples? Answer in maximum 50 words."


    result = llm_call(
        model="ollama/llama3.2:3b",
        prompt=prompt,
    )

    print("Printing the result")
    print(result)