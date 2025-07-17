import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import subprocess
import time
import uvicorn
import requests
from contextlib import asynccontextmanager
import anyio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "kimina-lean-server"))
from client.client import Lean4Client, batch_verify_proof
from utils.proof_utils import analyze

KIMINA_HOST = os.environ.get("KIMINA_HOST", "http://localhost:12332")
KIMINA_CLIENT = None
KIMINA_PROC = None

@asynccontextmanager
async def lifespan(app):
    global KIMINA_CLIENT, KIMINA_PROC
    # Start Kimina subprocess
    KIMINA_PROC = subprocess.Popen(
        ["python", "-m", "server"], 
        cwd=os.path.join(PARENT_DIR, "kimina-lean-server"),
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for it to be ready (blocking)
    KIMINA_CLIENT = await anyio.to_thread.run_sync(wait_for_kimina, KIMINA_HOST)
    print("Kimina client ready.")

    yield

    # Cleanup on shutdown
    print("Shutting down Kimina subprocess...")
    KIMINA_PROC.terminate()
    KIMINA_PROC.wait()
    print("Kimina subprocess shut down.")

app = FastAPI(lifespan=lifespan)


class Payload(BaseModel):
    context: str
    formal_statement: str
    answer: str
    problem_id: str

def _reconstruct_lean_executable(context, formal_statement, answer):
    return f'{context}\n{formal_statement}\n{answer}'

def _verify_kimina_requests(kimina_requests, timeout=30, batch_size=1):
    num_proc = os.cpu_count() or 16
    client = Lean4Client(base_url=KIMINA_HOST, disable_cache=True)
    
    result = batch_verify_proof(
        samples=kimina_requests,
        client=client,
        timeout=timeout,
        num_proc=num_proc,
        batch_size=batch_size,
    )
    
    return analyze(result)

@app.post("/check_problem_solution")
def check_solution(json: Payload, timeout=30):
    print("HERE")
    try:
        kimina_requests = [{
            "custom_id": json.problem_id,
            "proof": _reconstruct_lean_executable(json.context, json.formal_statement, json.answer)
        }]

        print(kimina_requests)
        print(KIMINA_CLIENT)

        response = KIMINA_CLIENT.verify(kimina_requests, timeout=timeout)
        print("RESPONSE", response)

        # "return_code": int,     # 0 if OK, -1 if timeout, -2 if unexpected server error. 
        # "score": float [0, 1],  # Reward between 0 and 1 inclusive 
        # "messages": list[str]   # Messages from the model (e.g. error messages.)

    except Exception as e:
        raise RuntimeError(f"Kimina verification failed: {e}")

@app.post("/batch_check_problem_solution")
def batch_check_solution(payloads: List[Payload], timeout=30, batch_size=1):
    try:
        kimina_requests = [{
            "custom_id": payload.problem_id,
            "proof": _reconstruct_lean_executable(payload.context, payload.formal_statement, payload.answer)
        } for payload in payloads]

        return _verify_kimina_requests(kimina_requests, timeout, batch_size)

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
    uvicorn.run("server:app", host="localhost", port=8007, reload=False)