import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import re
import subprocess
import time
import uvicorn
import requests
from contextlib import asynccontextmanager
import anyio
import sqlite3
import json as json_module  # Avoids conflict with FastAPI's `json` param
from fastapi import HTTPException

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "kimina-lean-server"))

from client.client import Lean4Client, batch_verify_proof
from utils.proof_utils import parse_client_response

KIMINA_HOST = os.environ.get("KIMINA_HOST", "http://localhost:12332")
KIMINA_CLIENT = None
KIMINA_PROC = None
LEAN_RL_DB_PATH = os.path.join(CURRENT_DIR, "lean_rl_db.db")

class Payload(BaseModel):
    solution: str
    problem_id: str

class Response(BaseModel):
    return_code: int     # 0 if OK, -1 if timeout, -2 if unexpected server error. 
    score: float         # [0, 1], Reward between 0 and 1 inclusive 
    messages: list[str]  # Messages from the model (e.g. error messages.)

def _create_sqlite_db(conn, cursor):
    for filename in os.listdir(CURRENT_DIR):
        if filename.endswith(".jsonl") and "sft" not in filename and "lean-rl-data" not in filename: # lean-rl-data is split into train/test, these are the parent files and redundant
            jsonl_path = os.path.join(CURRENT_DIR, filename)
            print(f"Loading {filename}...")

            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json_module.loads(line)
                        entry_id = entry.get("extra_info", {}).get("example_name")
                        if entry_id is not None:
                            cursor.execute(
                                "INSERT OR REPLACE INTO entries (id, data) VALUES (?, ?)",
                                (entry_id, json_module.dumps(entry))
                            )
                        else:
                            print(f"Skipping entry without example_name in {filename}")
                    except json_module.JSONDecodeError as e:
                        print(f"Skipping malformed line in {filename}: {e}")

    conn.commit()
    conn.close()
    print("All .jsonl files loaded into SQLite.")

def _get_context_and_formal_statement(problem_id):
    conn = sqlite3.connect(LEAN_RL_DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Fetch the stored entry by problem_id
    cursor.execute("SELECT data FROM entries WHERE id = ?", (problem_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Problem ID '{problem_id}' not found.")

    # Load the stored JSON entry
    try:
        stored_entry = json_module.loads(row[0])
        context = stored_entry["extra_info"]["context"]
        formal_statement = stored_entry["extra_info"]["formal_statement"]
        return context, formal_statement
    except Exception as e:
        print(f"Failed to parse stored JSON for problem ID {problem_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse stored JSON: {e}")

@asynccontextmanager
async def lifespan(app):
    global KIMINA_CLIENT, KIMINA_PROC

    # Start Kimina subprocess
    KIMINA_PROC = subprocess.Popen(
        ["python", "-m", "server"], 
        cwd=os.path.join(PARENT_DIR, "kimina-lean-server"),
        env=os.environ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for it to be ready (blocking)
    KIMINA_CLIENT = await anyio.to_thread.run_sync(wait_for_kimina, KIMINA_HOST)
    print("Kimina client ready.")

    # Load all .jsonl files into SQLite
    print("Connecting to SQLite database...")
    db_already_exists = os.path.exists(LEAN_RL_DB_PATH) # need to do this before connecting, otherwise it will create a new db

    conn = sqlite3.connect(LEAN_RL_DB_PATH)
    cursor = conn.cursor()

    # Create table
    cursor.execute("CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, data TEXT)")

    if not db_already_exists:
        _create_sqlite_db(conn, cursor)

    yield

    # Cleanup
    print("Shutting down Kimina subprocess...")
    KIMINA_PROC.terminate()
    KIMINA_PROC.wait()
    print("Kimina subprocess shut down.")


app = FastAPI(lifespan=lifespan)

def _reconstruct_lean_executable(context, formal_statement, solution):
    return f'{context}\n{formal_statement}\n{solution}'

def _verify_FVAPPS_response(spec, model_response):
    def _remove_whitespace(text):
        # Remove all whitespace (or use ' '.join(text.split()) to preserve single spaces)
        return re.sub(r'\s+', '', text.lower())

    # normalize by whitespace
    spec = _remove_whitespace(spec)
    model_response = _remove_whitespace(model_response)

    for segment in spec.split("sorry"):
        if segment not in model_response:
            return False # if any segment of the original spec is not in the solution, we return False
    return True # ok

@app.post("/check_problem_solution")
async def check_solution(json: Payload, timeout=30) -> Response:
    """
        Verify a single problem solution.

        Args:
            payload (Payload): Payload object containing problem_id and solution. The payload should contain:
                - problem_id: str
                - solution: str (the proof)
            timeout (int): Timeout for each request in seconds (default: 30).

        Returns:
            response: Response object containing:
                - return_code: int (0 if OK, -1 if timeout, -2 if unexpected server error)
                - score: float (between 0 and 1 inclusive)
                - messages: list[str] (messages from the model, e.g. error messages)
    """
    context, formal_statement = _get_context_and_formal_statement(json.problem_id)

    is_fvapps = bool(re.fullmatch(r"\d{4}", json.problem_id)) # FVAPPS problem IDs (and only these) are 4-digit numbers
    if is_fvapps:
        # For FVAPPS, we need to verify the response against the original spec (which is stored as context)
        is_valid_response = _verify_FVAPPS_response(context, json.solution)
        if not is_valid_response:
            return Response(
                return_code=0,
                score=0.0,
                messages=["The FVAPPS solution is missing part of the original spec."]
            )
        else:
            full_lean_executable = json.solution # for FVAPPS only, the solution is the full executable bc of the multiple sorries
    else:
        full_lean_executable = _reconstruct_lean_executable(context, formal_statement, json.solution)

    kimina_requests = [{
            "custom_id": json.problem_id,
            "proof": full_lean_executable
    }]

    try:
        print("Verifying Kimina...")

        result = await KIMINA_CLIENT.async_verify(kimina_requests, timeout=timeout)
        kimina_response = result['results'][0]
        
        print("Done verifying Kimina.")

        parse_score = parse_client_response(kimina_response)
        messages = kimina_response.get("messages", [])
        has_error = any(m.get("severity") == "error" for m in messages)

        return Response(
            return_code=0,
            score=parse_score["is_valid_no_sorry"],
            messages=[json_module.dumps(m) for m in messages]
        )
    except requests.exceptions.Timeout:
        return Response(return_code=-1, score=0.0, messages=["Timeout"])
    except Exception as e:
        print(f"Unexpected error: {e}")
        return Response(return_code=-2, score=0.0, messages=[str(e)])

@app.post("/batch_check_problem_solution")
def batch_check_solution(payloads: List[Payload], timeout=30, batch_size=1) -> List[Response]:
    """
        Batch verification for multiple problem solutions.

        Args:
            payloads (List[Payload]): List of Payload objects containing problem_id and solution. Each payload should contain:
                - problem_id: str
                - solution: str (the proof)
            timeout (int): Timeout for each request in seconds (default: 30).
            batch_size (int): Number of requests to send in a single batch (default: 1).

        Returns:
            responses: list of Response objects, one for each payload. Each Response contains:
                - return_code: int (0 if OK, -1 if timeout, -2 if unexpected server error)
                - score: float (between 0 and 1 inclusive)
                - messages: list[str] (messages from the model, e.g. error messages)
        
        If there is an error, a list containing a single Response with the error details is returned.
    """
    try:
        contexts, formal_statements = zip(*[_get_context_and_formal_statement(json.problem_id) for json in payloads])
        kimina_requests = [{
            "custom_id": payload.problem_id,
            "proof": _reconstruct_lean_executable(context, formal_statement, payload.solution)
        } for payload, context, formal_statement in zip(payloads, contexts, formal_statements)]

        num_proc = os.cpu_count() or 16
        
        kimina_results = batch_verify_proof(
            samples=kimina_requests,
            client=KIMINA_CLIENT,
            timeout=timeout,
            num_proc=num_proc,
            batch_size=batch_size,
        )

        responses = []
        for result in kimina_results:
            kimina_response = result['response']
            score = parse_client_response(kimina_response)["is_valid_no_sorry"]
            messages = kimina_response.get("messages", [])
            
            responses.append(Response(
                return_code=0,
                score=score,
                messages=[json_module.dumps(m) for m in messages]  # stringify each dict
            ))
        return responses
    except requests.exceptions.Timeout:
        return [Response(return_code=-1, score=0.0, messages=["Timeout"])]
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [Response(return_code=-2, score=0.0, messages=[str(e)])]

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
    uvicorn.run("server:app", host="0.0.0.0", port=8007, reload=False)