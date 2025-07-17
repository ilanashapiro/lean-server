import os
import sys
import time
import json

import server

from datasets import load_dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "kimina-lean-server"))
from client.client import Lean4Client, batch_verify_proof
from utils.proof_utils import analyze

kimina_client = Lean4Client(base_url=server.KIMINA_HOST)

def process_jsonl_file(path, n_samples) -> list:
    payloads = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                raw = json.loads(line)
                payload = {
                    "problem_id": raw["extra_info"]["example_name"],
                    "answer": raw["extra_info"]["ground_truth"],
                    "context": raw["extra_info"]["context"],
                    "formal_statement": raw["extra_info"]["formal_statement"]
                }

                # this is from LeanDockerWrapper/kimina-lean-server/benchmark.py for the real example
                # dataset = load_dataset("Goedel-LM/Lean-workbook-proofs", split="train")
                # dataset = dataset.select(range(24999, 25000))

                # samples = [
                #     {"custom_id": sample["problem_id"], "proof": sample["full_proof"]}
                #     for sample in dataset
                # ]

                # print("SAMPLE", samples[0])

                payload_dict = server.Payload(**payload_dict)
                proof = server._reconstruct_lean_executable(
                        payload.context, payload.formal_statement, payload.answer
                    )
                # print(proof)
                # kimina_request = [{
                #     "custom_id": samples[0]["custom_id"],
                #     "proof": samples[0]["proof"]
                # }]
                # kimina_request = {
                #     "custom_id": payload.problem_id,
                #     # "proof": 'import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n\ntheorem lean_workbook_plus_64000 (x y : ℝ) (hx : x = 2) : x^y - x = y^2 - y ↔ 2^y - 2 = y^2 - y   := by\n  \n  constructor\n  intro h\n  rw [hx] at h\n  exact h\n  intro h\n  rw [hx]\n  exact h\n'
                #     # "proof": 'import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\ntheorem lean_workbook_plus_64000 (x y : ℝ) (hx : x = 2) : x^y - x = y^2 - y ↔ 2^y - 2 = y^2 - y   := by\n  constructor\n  intro h\n  rw [hx] at h\n  exact h\n  intro h\n  rw [hx]\n  exact h'
                #     "proof": proof
                # }

                payloads.append({"custom_id": payload_dict["problem_id"], "proof": proof})

                # print(kimina_request)
                # start_time = time.time()
                # responses = kimina_client.verify(kimina_request, timeout=30)
                # elapsed_time = time.time() - start_time
                # print(f"Kimina response time: {elapsed_time:.2f} seconds")
                # print(responses)
                if len(payloads) >= n_samples:
                    break
            except Exception as e:
                print(f"[Line {line_num}] Error: {e}")
                continue
    return payloads

if __name__ == "__main__":
    jsonl_path = os.path.join(CURRENT_DIR, "lean-train-rl-data-Lean-Workbook.jsonl")

    n = 1
    timeout = 30
    batch_size = 1
    num_proc = os.cpu_count() or 16
    url = "http://localhost:12332"

    payloads = process_jsonl_file(jsonl_path, n)

    result = batch_verify_proof(
        samples=payloads,
        client=kimina_client,
        timeout=timeout,
        num_proc=num_proc,
        batch_size=batch_size,
    )

    analyze(result)