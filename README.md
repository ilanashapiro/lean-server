# Lean Server
Lean 4 verification server based on [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)

## Generating SFT and RL annotated datasets
You can run the `gen_rl.py` script to generate annotated RL data from [Lean-Workbook](https://huggingface.co/datasets/internlm/Lean-Workbook)/[Goedel LM](https://huggingface.co/datasets/Goedel-LM/Lean-workbook-proofs), [FVAPPS](https://huggingface.co/datasets/quinn-dougherty/fvapps), [Verina](https://huggingface.co/datasets/sunblaze-ucb/verina), and [MiniF2F](https://huggingface.co/datasets/Tonic/MiniF2F). 
Train/val splits come from Lean-Workbook/Goedel LM and FVAPPS, and testing splits come from Verina and MiniF2F.
We restrict the examples from Lean-Workbook to be those that have ground truth proofs in Goedel LM, since it is not clear if the remaining examples in Lean-Workbook are provable.
Some of the examples in Verina have ground truth, but most do not. FVAPPS and MiniF2F do not have ground truth.

The resulting .jsonl files will all be saved to the root directory (you can the SFT files and the unsplit FVAPPS/Lean Workbook files there. The latter 2 files are then further split into train and val). The splits used for RL train/val/test were manually moved inside wrapper_server so that they are baked into the Docker image and stored in the server's database.

## Building the docker image
Please allow for ~10min to build the image.
```
docker build -t lean-server .
```

## Verification Server
1. Starting the verification server (run from the root directory).
```
docker run -p 8007:8007 lean-server uvicorn wrapper_server.server:app --host 0.0.0.0 --port 8007
```
Wait until the startup is complete. You should see something like this:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2025-07-17 23:32:33.876 | Level 40 | tenacity.before_sleep:log_it:65 - Retrying client.client.Lean4Client._query.<locals>.query_with_retries in 1.0 seconds as it raised ClientConnectorError: Cannot connect to host localhost:12332 ssl:default [Connect call failed ('127.0.0.1', 12332)].
2025-07-17 23:32:34.886 | Level 40 | tenacity.before_sleep:log_it:65 - Retrying client.client.Lean4Client._query.<locals>.query_with_retries in 2.0 seconds as it raised ClientConnectorError: Cannot connect to host localhost:12332 ssl:default [Connect call failed ('127.0.0.1', 12332)].
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8007 (Press CTRL+C to quit)
```

2. Calling the verification server for single examples

Once the verification engine is started, you can call it (from the host machine) to verify single examples through requests like the following:
```
curl --request POST \
  --url http://localhost:8007/check_problem_solution \
  --header 'Content-Type: application/json' \
  --data '{
		"problem_id": "lean_workbook_plus_2",
		"answer": "  \n  constructor\n  intro hx\n  constructor\n  nlinarith [hx]\n  nlinarith [hx]\n  rintro ⟨h1, h2⟩\n  nlinarith"
}' | jq
```

Note that `problem_id` needs to correspond to something an `example_name` that's actually in the Lean annotated datasets inside `wrapper_server` (these are in an SQLite database in the image). This database is queried to get the remainder of the Lean file needed to make the answer into a complete Lean executable that Kimina can check.
This should give you the following successful response:
```
{
  "return_code": 0,
  "score": 1,
  "messages": []
}
```

2. Calling the verification server for batch verification of multiple examples

You can also call the the server (from the host machine) to batch verify a list of examples through requests like the following:
```
curl --request POST \
  --url http://localhost:8007/batch_check_problem_solution \
  --header 'Content-Type: application/json' \
  --data '[
    {
      "problem_id": "lean_workbook_plus_2",
		  "answer": "  \n  constructor\n  intro hx\n  constructor\n  nlinarith [hx]\n  nlinarith [hx]\n  rintro ⟨h1, h2⟩\n  nlinarith"
    },
    {
      "problem_id": "verina_basic_78",
		  "answer": "  unfold MultipleReturns_postcond MultipleReturns\n  simp"
    }
  ]' | jq
```

This should give you the following successful response:
```
[
  {
    "return_code": 0,
    "score": 1,
    "messages": [
      "{\"severity\": \"warning\", \"pos\": {\"line\": 4, \"column\": 29}, \"endPos\": {\"line\": 4, \"column\": 30}, \"data\": \"unused variable `x`\\nnote: this linter can be disabled with `set_option linter.unusedVariables false`\"}",
      "{\"severity\": \"warning\", \"pos\": {\"line\": 4, \"column\": 39}, \"endPos\": {\"line\": 4, \"column\": 40}, \"data\": \"unused variable `y`\\nnote: this linter can be disabled with `set_option linter.unusedVariables false`\"}",
      "{\"severity\": \"warning\", \"pos\": {\"line\": 7, \"column\": 41}, \"endPos\": {\"line\": 7, \"column\": 50}, \"data\": \"unused variable `h_precond`\\nnote: this linter can be disabled with `set_option linter.unusedVariables false`\"}",
      "{\"severity\": \"warning\", \"pos\": {\"line\": 13, \"column\": 72}, \"endPos\": {\"line\": 13, \"column\": 81}, \"data\": \"unused variable `h_precond`\\nnote: this linter can be disabled with `set_option linter.unusedVariables false`\"}"
    ]
  },
  {
    "return_code": 0,
    "score": 1,
    "messages": []
  }
]
```

3. **You should follow [test_wrapper_server.py](test_wrapper_server.py) for an example of calling the verification server programatically**. It shows how to call the server in on both single examples and batch verification.
