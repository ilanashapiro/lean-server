import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import subprocess
import time
import uvicorn
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "kimina-lean-server"))
from client.client import Lean4Client, batch_verify_proof
from utils.proof_utils import analyze

KIMINA_HOST = os.environ.get("KIMINA_HOST", "http://localhost:12332")
KIMINA_CLIENT = None

app = FastAPI()

class Payload(BaseModel):
    context: str
    formal_statement: str
    answer: str
    problem_id: str

def reconstruct_lean_executable(context, formal_statement, answer):
    return f'{context}\n{formal_statement}\n{answer}'

@app.post("/check_problem_solution")
def check_solution(payload: Payload):
    try:
        kimina_request = [{
            "custom_id": payload.problem_id,
            "proof": reconstruct_lean_executable(payload.context, payload.formal_statement, payload.answer)
        }]
        start_time = time.time()
        responses = client.verify(kimina_request, timeout=30)
        elapsed_time = time.time() - start_time
        print(f"Kimina response time: {elapsed_time:.2f} seconds")

        # "return_code": int,     # 0 if OK, -1 if timeout, -2 if unexpected server error. 
        # "score": float [0, 1],  # Reward between 0 and 1 inclusive 
        # "messages": list[str]   # Messages from the model (e.g. error messages.)

        # if responses and responses[0].get("result") == "success":
        #     return {"return_code": 0, "score": 1.0}
        # else:
        #     return {"return_code": 1, "score": 0.0}

    except Exception as e:
        print(f"Kimina call failed: {e}")
        return {"return_code": -2, "score": 0, "messages": [str(e)]}

@app.post("/batch_check_problem_solution")
def batch_check_solution(payloads: List[Payload], timeout = 30, batch_size = 1):
    try:
        kimina_requests = [{
            "custom_id": payload.problem_id,
            "proof": reconstruct_lean_executable(payload.context, payload.formal_statement, payload.answer)
        } for payload in payloads]

        timeout = 30
        batch_size = 1
        num_proc = os.cpu_count() or 16
        
        result = batch_verify_proof(
            samples=kimina_requests,
            client=KIMINA_CLIENT,
            timeout=timeout,
            num_proc=num_proc,
            batch_size=batch_size,
        )

        analyze(result)

    except Exception as e:
        raise RuntimeError(f"Kimina batch verification failed: {e}")
    
def wait_for_kimina(host, timeout=30):
    try:
        client = Lean4Client(base_url=host)
        kimina_request = [{
            "custom_id": "1234",
            "proof": "#check Nat"
        }]
        # e.g. {'results': [{
        #           'custom_id': '1234', 
        #           'error': None, 
        #           'response': {
        #               'messages': [{
        #                   'severity': 'info', 
        #                   'pos': {'line': 1, 'column': 0}, 
        #                   'endPos': {'line': 1, 'column': 6}, 
        #                   'data': 'Nat : Type'}], 
        #       'env': 0, 
        #       'time': 0.6712229251861572}}]}
        responses = client.verify(kimina_request, timeout=timeout)
        if not responses['results'][0]['error']:
            print("Kimina server is ready.")
            return client
    except Exception as e:
        raise RuntimeError(f"Kimina server failed to start with exception: {e}")

if __name__ == "__main__":
    kimina_proc = subprocess.Popen(
        ["python", "-m", "server"], 
        cwd=f"{PARENT_DIR}/kimina-lean-server",
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait until Kimina is ready
        KIMINA_CLIENT = wait_for_kimina(KIMINA_HOST)

        # Start your wrapper FastAPI server
        uvicorn.run("server:app", host="localhost", port=8007, reload=False)

    finally:
        kimina_proc.terminate()
        kimina_proc.wait()


# docker run -p 8000:8000 --rm lean-server uvicorn server:app --host 0.0.0.0 --port 8000