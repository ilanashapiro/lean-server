import subprocess
import time
import socket
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import sys
import json
import requests
import logging
import multiprocessing
import atexit

from tqdm import tqdm
from torch.utils.data import Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.append("kimina-lean-server")
from client import Lean4Client

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/home/t-ilshapiro/CoqStoq/fstarcoq-qwq-32b-singleturn-sft") # path that points to the directory with the model name (e.g. fstarcoq-qwq-32b...)
    parser.add_argument("--sample_n", type=int, default=1) # how many times we sample for each prompt (i.e. sample on same input)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=2)
    args = parser.parse_args()

    # Load validation data
    print("Loading validation data...")
    valid_data = []
    with open("coq-test-data-with-responses.jsonl") as file:
        for line in file:
            valid_data.append(json.loads(line))
    if args.debug:
        valid_data = valid_data[:100]
    else:
        valid_data = valid_data[:600] # can use c. 700 for test benchmark

    # Load tokenizer and vLLM engine
    print(f"Loading tokenizer and checkpoint from {args.model_name}... ", end="")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    llm = LLM(model=args.model_name, dtype="bfloat16", max_model_len=16384, tensor_parallel_size=args.num_gpus)

    # Prepare prompts
    print("Preparing prompts...")
    prompts = []
    prompt_to_index = []  # (datum_idx, sample_idx)
    for datum_idx, datum in enumerate(tqdm(valid_data)):
        prompt = datum["user_prompt"]
        if len(tokenizer(prompt).input_ids) > 8192:
            continue
        for sample_idx in range(args.sample_n):
            prompts.append(prompt)
            prompt_to_index.append((datum_idx,sample_idx))

    # Generate with vLLM
    print(f"Sampling responses... {args.sample_n} samples per prompt, temp={args.temperature}")
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=16384, n=1)
    outputs = llm.generate(prompts, sampling_params)
    print("Done sampling")

    # Organize responses into valid_data
    for datum in valid_data:
        datum["model_generated_response"] = [] # length of this list will be sample_n

    for output, (datum_idx, _) in zip(outputs, prompt_to_index):
        response = output.outputs[0].text
        if "<answer>" in response and "</answer>" in response:
            valid_data[datum_idx]["model_generated_response"].append(response) # recall datum_idx is the line number in the jsonl file
    
    output_path = "coq-test-data-with-responses.jsonl"

    with open(output_path, "a") as f:
        for datum in valid_data:
            json.dump(datum, f)
            f.write("\n")

    print(f"Saved {len(valid_data)} entries to {output_path}")

    # Evaluation
    pass_n_cnt = [0 for _ in range(args.sample_n)]
    results = []
    print("Evaluating model outputs...")
    
    tasks = []
    for datum in valid_data:
        datum_id = datum["extra_info"]["example_name"]
        prompt = datum["prompt"]["content"]
        model_generated_responses = datum["model_generated_response"]
        answers = [response.split("<answer>")[1].split("</answer>")[0] for response in model_generated_responses]
        pass_flag = False
        
        kimina_requests = [{"proof:": answer, "custom_id": datum_id} for answer in answers]
        client = Lean4Client(base_url="http://127.0.0.1:12332")
        responses = client.verify(kimina_requests, timeout=30)
        
        for i, response in enumerate(responses):
            # INSPECT KIMINA OUTPUT FORMAT WHEN WE HAVE DATA

            if result_value:
                pass_flag = True
            pass_n_cnt[i] += 1 if pass_flag else 0

            if args.debug:
                # print("Prompt:")
                # print(prompt)
                # print("Model Output:")
                # print(model_generated_responses[i])
                print("Passed?", result_value)
                # if not result_value:
                #     print(errormsg)
                print()
            else:
                results.append({
                    "example_name": f"{split}_{index}",
                    "prompt": prompt,
                    "model_output": model_generated_responses[i],
                    "result": result_value,
                    "errormsg": errormsg
                })

    print("")
    print("Total data:", len(valid_data))
    print("Pass@n:", [x / len(valid_data) for x in pass_n_cnt])

    if not args.debug:
        with open("coq_valid_test.json", "w") as file:
            json.dump(results, file, indent=4)
        print("Results saved to coq_valid_test.json")