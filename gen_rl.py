import json, sys, time, subprocess, re

from pathlib import Path
from gen_sft import combine_to_jsonl
from datasets import load_dataset
from enum import Enum

class DataSource(Enum):
    VERINA = "Verina"
    FVAPPS = "FVAPPS"
    MINIF2F = "MiniF2F"
    LEAN_WORKBOOK = "Lean-Workbook"

DATA_SOURCE_ENUM = DataSource.MINIF2F
DATA_SOURCE = DATA_SOURCE_ENUM.value

USER_PROMPT_FORMAT_VERINA = """I need to solve the following task in Lean 4: \n```\n{informal_statement}\n```\n#####\n\n More formally, I need to prove the following theorem in Lean 4: \n```\n{theorem}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""
USER_PROMPT_FORMAT_FVAPPS = """The question I need to answer in Lean 4 is as follows: \n```\n{question}\n```\n#####\n\nTo answer this question, you need to replace the {num_sorries} \"sorry\" keywords in the following Lean spec with Lean proofs that make the resulting Lean executable work: \n```\n{spec}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""
USER_PROMPT_MINIF2F = """I need to solve the following task in Lean 4: \n```\n{informal_statement}\n```\n#####\n\n More formally, I need to prove the following theorem in Lean 4: \n```\n{theorem}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""

match DATA_SOURCE_ENUM:
    case DataSource.VERINA:
        DATASET = load_dataset("sunblaze-ucb/verina", split="train") # keys: ['id', 'description', 'lean_code', 'signature', 'metadata', 'tests', 'reject_inputs', 'difficulty']
        SPLIT = "test"
    case DataSource.FVAPPS:
        DATASET = load_dataset("quinn-dougherty/fvapps", split="train") # keys: ['apps_id', 'apps_question', 'spec', 'units', 'sorries', 'apps_difficulty', 'assurance_level']
        SPLIT = "train-rl" # set as train for now, then randomly sample ~200 for validation after
    case DataSource.MINIF2F:
        DATASET = load_dataset("Tonic/MiniF2F", split="train") # keys: ['name', 'split', 'informal_prefix', 'formal_statement', 'goal', 'header']
        SPLIT = "test"
    case DataSource.LEAN_WORKBOOK:
        DATASET = load_dataset("internlm/Lean-Workbook", split="train") # keys: ['id', 'status', 'tactic', 'state_before', 'state_after', 'natural_language_statement', 'answer', 'formal_statement']
        SPLIT = "train-rl"
    case _:
        raise ValueError(f"Unsupported data source: {DATA_SOURCE_ENUM}")

SAVE_LOC = Path(f"lean-{SPLIT}-data-{DATA_SOURCE}")
SAVE_LOC.mkdir(parents=True, exist_ok=True)
OUT_PATH_JSONL = SAVE_LOC.with_suffix(".jsonl")

TRAIN_DATA = list(DATASET) # this should load in order, can write sorting function if we run into issues
NUM_EXAMPLES = len(TRAIN_DATA)
START_INDEX = 0#sum(1 for f in SAVE_LOC.iterdir() if f.is_file())

def parse_verina_lean_code(lean_code: str) -> dict[str, str]:
    lines = lean_code.splitlines()

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep the benchmark start marker line, don't remove it
        if stripped.startswith("--") and stripped != "-- !benchmark @start proof":
            continue
        cleaned_lines.append(line)

    cleaned_code = "\n".join(cleaned_lines)
    cleaned_code = re.sub(r"\n{3,}", "\n\n", cleaned_code) # remove multiple newlines

    thm_idx = cleaned_code.find("theorem ")
    if thm_idx == -1:
        raise ValueError("No 'theorem ' found in the provided Lean code.")

    context = cleaned_code[:thm_idx]
    remaining_lines = cleaned_code[thm_idx:].splitlines()

    benchmark_start_idx = next(
        (i for i, line in enumerate(remaining_lines) if line.strip() == "-- !benchmark @start proof"),
        None
    )
    if not benchmark_start_idx:
        raise ValueError("No benchmark start marker found in the provided Lean code.")
    
    formal_statement = "\n".join(remaining_lines[:benchmark_start_idx])
    ground_truth = "\n".join(remaining_lines[:benchmark_start_idx] + remaining_lines[benchmark_start_idx + 1:]) if all("sorry" not in line for line in remaining_lines) else None

    return {
        "context": context,
        "formal_statement": formal_statement,
        "ground_truth": ground_truth,
    }
    # print(res["context"])
    # print("#############################################################################################")
    # print(res["formal_statement"])
    # print("#############################################################################################")
    # print(res["ground_truth"])
    # sys.exit(0)


def augment():
    print(f"Starting RL augmentation from index {START_INDEX} for {NUM_EXAMPLES} examples in data source {DATA_SOURCE}.")
    augmented_count = 0

    for datum in TRAIN_DATA[START_INDEX:START_INDEX + NUM_EXAMPLES]:
        match DATA_SOURCE_ENUM:
            case DataSource.VERINA: # keys: ['id', 'description', 'lean_code', 'signature', 'metadata', 'tests', 'reject_inputs', 'difficulty']
                parsed_lean_code = parse_verina_lean_code(datum['lean_code'])
                user_prompt = USER_PROMPT_FORMAT_VERINA.format(informal_statement=datum["description"], theorem=parsed_lean_code["formal_statement"], file_context=parsed_lean_code["context"])
                datum_id = datum["id"]
                ground_truth = parsed_lean_code["ground_truth"]
            case DataSource.FVAPPS: # keys: ['apps_id', 'apps_question', 'spec', 'units', 'sorries', 'apps_difficulty', 'assurance_level']
                user_prompt = USER_PROMPT_FORMAT_FVAPPS.format(question=datum["apps_question"], num_sorries=datum['sorries'], spec=datum["spec"])
                datum_id = datum["apps_id"]
                ground_truth = None
            case DataSource.MINIF2F: # keys: ['name', 'split', 'informal_prefix', 'formal_statement', 'goal', 'header']
                user_prompt = USER_PROMPT_MINIF2F.format(informal_statement=datum["informal_prefix"], theorem=datum["formal_statement"], file_context=datum["header"])
                datum_id = datum["name"]
                ground_truth = None
            case DataSource.LEAN_WORKBOOK: # keys: ['id', 'status', 'tactic', 'state_before', 'state_after', 'natural_language_statement', 'answer', 'formal_statement']
                user_prompt = """"""                
                datum_id = datum["name"]
                ground_truth = None # REPLACE WITH GOEDEL PROOF
            case _:
                raise ValueError(f"Unsupported data source: {DATA_SOURCE_ENUM}")

        entry = {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": user_prompt}],
            "ability": "programming",
            "reward_model": {"style": "execution", "ground_truth": ground_truth},
            "extra_info": {
                "language": "Lean", # "F*" "Coq" "Lean"
                "example_name": datum_id,
                "prompt": user_prompt,
                "ground_truth": ground_truth,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "tools/execute_fstar": {
                        "example_name": datum_id
                    }
                },
            }
        }

        # ──────────────────────────────────────────────────────────────
        # Write ONE file per example
        #   e.g.  sft-data/train-sft_226.json
        # ──────────────────────────────────────────────────────────────
        out_path = SAVE_LOC / f"{datum_id}.json"
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(entry, f_out, ensure_ascii=False, indent=2)

        augmented_count += 1
        print(f"{augmented_count+START_INDEX}/{NUM_EXAMPLES} done  ->  {SAVE_LOC}", file=sys.stderr)

    print(f"Wrote {augmented_count} files to “{SAVE_LOC}”.")
    
if __name__ == "__main__":
    augment()
    combine_to_jsonl(SAVE_LOC, OUT_PATH_JSONL)