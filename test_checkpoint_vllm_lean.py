import subprocess
import time
import socket
import argparse
import os
import sys
import json
import requests # install
import logging
import multiprocessing
import atexit
import aiohttp # install
import asyncio
from collections import defaultdict

from tqdm import tqdm # install
from torch.utils.data import Dataset # install
from vllm import LLM, SamplingParams # install
from transformers import AutoTokenizer # install
# from example import check_ground_truth, Task, CheckingResult, VerificationResult, ErrorResult

async def evaluate_single_example(session, url, payload):
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            return result.get("return_code", -2), result.get("score", 0.0), result.get("messages", "Unknown error")
    except Exception as e:
        return -2, 0.0, str(e.__class__) + ": " + str(e)

async def evaluate_batch(batch, url, args):
    tasks = []
    async with aiohttp.ClientSession() as session:
        for datum, i, response, name in batch:
            try:
                code = response.split("<answer>")[1].split("</answer>")[0]
                payload = {
                    "solution": code,
                    "problem_id": name
                }
                
                task = evaluate_single_example(session, url, payload)
                tasks.append((datum, i, response, task))
            except IndexError:
                # Handle case where response doesn't contain proper tags
                continue
        
        results = []
        for datum, i, response, task in tasks:
            try:
                return_code, score, errormsg = await task
                result = return_code == 0 and score == 1.0
            except Exception as e:
                # Timeout, cancelled, etc
                result = False
                errormsg = str(e.__class__) + ": " + str(e)
                
            if args.debug:
                print("Example name:", datum["extra_info"]["example_name"])
                print("Prompt:")
                print(datum["prompt"][0]["content"])
                print("Model Output:")
                print(response)
                print("Passed?", result)
                if not result:
                    print(errormsg)
                print()
            results.append({
                "example_name": datum["extra_info"]["example_name"],
                "prompt": datum["prompt"][0]["content"],
                "model_output": response,
                "result": result,
                "errormsg": errormsg
            })
        
        return results

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sample_n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test-set", type=str, default="Verina", choices=["Verina", "MiniF2F"])
    args = parser.parse_args()

    # Load validation data
    print("Loading validation data...")
    valid_data = []
    with open(f"wrapper_server/lean-test-data-{args.test_set}.jsonl") as file:
        for line in file:
            valid_data.append(json.loads(line))
    if args.debug:
        valid_data = valid_data[:2]
    else:
        valid_data = valid_data[:600]  # can use c. 700 for test benchmark

    url = os.environ.get("LEAN_VERIFIER_SERVER_HOST", "http://localhost:8007") + "/check_problem_solution"

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
        prompt = datum["prompt"][0]["content"]
        if len(tokenizer(prompt).input_ids) > 8192:
            continue
        for sample_idx in range(args.sample_n):
            prompts.append(prompt)
            prompt_to_index.append((datum_idx, sample_idx))

    # Generate with vLLM
    print(f"Sampling responses... {args.sample_n} samples per prompt, temp={args.temperature}")
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=16384, n=1)
    outputs = llm.generate(prompts, sampling_params)
    print("Done sampling")

    # Organize responses into valid_data
    for datum in valid_data:
        datum["model_generated_response"] = []  # length of this list will be sample_n

    for output, (datum_idx, _) in zip(outputs, prompt_to_index):
        response = output.outputs[0].text
        if "<answer>" in response and "</answer>" in response:
            valid_data[datum_idx]["model_generated_response"].append(response)

    output_path = f"lean-test-data-with-responses-{args.test_set}.jsonl"

    with open(output_path, "w") as f:
        for datum in valid_data:
            json.dump(datum, f)
            f.write("\n")

    print(f"Saved {len(valid_data)} entries to {output_path}")

    # Evaluation using batched async approach
    BATCH_SIZE = args.batch_size
    all_results = []

    batches = []
    current_batch = []

    # Prepare all evaluation items
    for datum in valid_data:
        datum_id = datum["extra_info"]["example_name"]
        for i, response in enumerate(datum["model_generated_response"]):
            current_batch.append((datum, i, response, datum_id))

            if len(current_batch) == BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []
    
    # Add the remaining items as the last batch
    if current_batch:
        batches.append(current_batch)

    # Process batches with progress bar
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        batch_results = asyncio.run(evaluate_batch(batch, url, args))
        all_results.extend(batch_results)

    # Calculate Pass@n
    pass_n = defaultdict(list)
    for res in all_results:
        pass_n[res["example_name"]].append(res["result"])

    print("")
    print("Total data:", len(valid_data))
    for i in range(args.sample_n):
        pass_count = sum(1 for results in pass_n.values() if any(results[:i+1]))
        print("PASS COUNT:", pass_count)
        print(f"Pass@{i+1}:", pass_count / len(valid_data))

    if not args.debug:
        with open(f"lean_valid_test_{args.test_set}.json", "w") as file:
            json.dump(all_results, file, indent=4)