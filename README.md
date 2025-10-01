# Finetuning Llama-2.3-1B using LoRA on Bloomberg Terminal System data

Problem Statement:
I noticed that LLM were performing poorly when I had specific questions related to navigation or understanding the Bloomberg Terminal system. Hence, I decided to finetune a model on the Bloomberg Terminal system data to improve its performance on such queries.

Base Model: llama3.2:1b
Finetuning Method: LoRA (Low-Rank Adaptation)
Framework: Hugging Face Transformers and PEFT (Parameter-Efficient Fine-Tuning)
Bigger Model: llama3.1:8b (For generating QA pairs)

Ollama: To run models locally
LangFlow: For visual model interactions

Baseline for Evaluation: We added set of 5 quetions to evaluate the model's performance before and after finetuning. Used Langflow to evaluate the model's performance on these questions. Documented the results in evaluations_questions.csv


Ollama:
| Command                        | Description                                       |
|--------------------------------|---------------------------------------------------|
| `ollama stop`                  | Stop all running Ollama models                    |
| `ollama ps`                    | List currently running models                     |
| `ollama stop <model_name>`     | Stop a specific model                             |
| `pkill ollama`                 | Force kill Ollama process                         |
| `sudo systemctl stop ollama`   | Stop Ollama service (if installed as service)     |
| `sudo systemctl start ollama`  | Start Ollama service (if installed as service)    |
| `sudo systemctl restart ollama`| Restart Ollama service (if installed as service)  |
| `ollama pull <model_name>`     | Pull a model from the Ollama registry             |
| `ollama run <model_name>`      | Run a model locally                               |
| `ollama list`                  | List all available models in the Ollama registry  |
| `ollama delete <model_name>`   | Delete a model from local storage                 |
|--------------------------------|---------------------------------------------------|


To use this fine tuned model, ensure you have Ollama installed and follow these steps:
1. Pull this repository to your local machine.
2. Open a terminal and navigate to the directory where you cloned the repository.
3. Run the following command to start the fine-tuned model:
    `ollama create bloomberg-llama -f ./Modelfine'
4. Once the model is created, you can run it using:
    `ollama run bloomberg-llama`

You can also use langflow to interact with the model visually. Just set the model name to `bloomberg-llama` in your langflow configuration.
Steps:
1. Start Langflow server:
    `langflow run`
2. In Ollama component in you configuration, and set the model name to `bloomberg-llama`.