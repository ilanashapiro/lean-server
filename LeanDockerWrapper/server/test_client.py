import test_kimina
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
                    "context": raw["extra_info"]["context"],
                    "formal_statement": raw["extra_info"]["formal_statement"]
                }

                payload = server.Payload(**payload_dict)
                payloads.append(payload)
                if len(payloads) >= n_samples:
                    break
            except Exception as e:
                print(f"[Line {line_num}] Error: {e}")
                continue
    return payloads

if __name__ == "__main__":
    url = "http://localhost:8007/check_problem_solution"
    jsonl_path = os.path.join(test_kimina.CURRENT_DIR, "lean-train-rl-data-Lean-Workbook.jsonl")

    payload = process_jsonl_file(jsonl_path, 1)[0]

    try:
        response = requests.post(url, json=payload.model_dump(), timeout=15000)
        response.raise_for_status()
        resp_json = response.json()
        # return_code = resp_json.get("return_code", -2)
        # if return_code != 0:
        #     print(f"Request failed with return code {return_code} for example {payload.problem_id}")
        # else:
        #     print(f"Response succeeded with score {resp_json.get('score', -1.0)}")
    except Exception as e:
        print(f"Request failed for example {payload.problem_id}: {e}, returning -1")