import LeanDockerWrapper.server.test_kimina_MESSY as test_kimina_MESSY
import os, sys, json
import requests
import server

def process_jsonl_file(path, n_samples) -> list:
    payloads = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                raw = json.loads(line)
                payload_dict = {
                    "problem_id": raw["extra_info"]["example_name"],
                    "answer": raw["extra_info"]["ground_truth"],
                }

                payload = server.Payload(**payload_dict)
                payloads.append(payload)
                if len(payloads) >= n_samples:
                    break
            except Exception as e:
                print(f"[Line {line_num}] Error: {e}")
                continue
    return payloads

def test_single_examples(payloads, url="http://localhost:8007/check_problem_solution"):
    for payload in payloads:
        try:
            response = requests.post(url, json=payload.model_dump(), timeout=15000)
            response.raise_for_status()
            resp_json = response.json()
            print(f"Response for example {payload.problem_id}: {resp_json['return_code']}, Score: {resp_json['score']}")
        except Exception as e:
            print(f"Request failed for example {payload.problem_id}: {e}, returning -1")

def test_batch_examples(payloads, url="http://localhost:8007/batch_check_problem_solution"):
    try:
        response = requests.post(url, json=[payload.model_dump() for payload in payloads], timeout=15000)
        response.raise_for_status()
        resp_json = response.json()
        for payload, result in zip(payloads, resp_json):
            print(f"Response for example {payload.problem_id}: {result['return_code']}, Score: {result['score']}")
    except Exception as e:
        print(f"Request failed for batch: {e}, returning -1")

if __name__ == "__main__":
    jsonl_path = os.path.join(test_kimina_MESSY.CURRENT_DIR, "lean-train-rl-data-Lean-Workbook.jsonl")

    payloads = process_jsonl_file(jsonl_path, 50)[0:]
    # test_single_examples(payloads)
    test_batch_examples(payloads)