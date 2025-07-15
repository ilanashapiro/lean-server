import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

sys.path.append("kimina-lean-server")
from client import Lean4Client

def reconstruct_lean_executable(context, formal_statement, answer):
    return f"{context}\n{formal_statement}\n{answer}"

app = FastAPI()

KIMINA_HOST = os.environ.get("KIMINA_HOST", "http://host.docker.internal:12332")

class Payload(BaseModel):
    context: str
    formal_statement: str
    answer: str
    problem_id: str

@app.post("/check_problem_solution")
def check_solution(payload: Payload):
    try:
        client = Lean4Client(base_url=KIMINA_HOST)
        kimina_request = [{
            "custom_id": payload.problem_id,
            "proof": reconstruct_lean_executable(payload.context, payload.formal_statement, payload.answer)
        }]
        responses = client.verify(kimina_request, timeout=30)

        if responses and responses[0].get("result") == "success":
            return {"return_code": 0, "score": 1.0}
        else:
            return {"return_code": 1, "score": 0.0}

    except Exception as e:
        print(f"Kimina call failed: {e}")
        return {"return_code": -1, "score": -1.0}