import json, sys, re, random, os

from pathlib import Path
from gen_sft import combine_to_jsonl
from datasets import load_dataset
from enum import Enum

class DataSource(Enum):
    VERINA = "Verina"
    FVAPPS = "FVAPPS"
    MINIF2F = "MiniF2F"
    LEAN_WORKBOOK = "Lean-Workbook"

DATA_SOURCE_ENUM = DataSource.FVAPPS
DATA_SOURCE = DATA_SOURCE_ENUM.value

USER_PROMPT_VERINA = """I need to solve the following task in Lean 4: \n```\n{informal_statement}\n```\n#####\n\n More formally, I need to prove the following theorem in Lean 4: \n```\n{formal_statement}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""
USER_PROMPT_FVAPPS = """The question I need to answer in Lean 4 is as follows: \n```\n{question}\n```\n#####\n\nTo answer this question, you need to replace the {num_sorries} \"sorry\" keywords in the following Lean 4 spec with Lean 4 proofs that make the resulting Lean 4 executable work: \n```\n{spec}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""
USER_PROMPT_MINIF2F = """I need to solve the following task in Lean 4: \n```\n{informal_statement}\n```\n#####\n\n More formally, I need to prove the following theorem in Lean 4: \n```\n{formal_statement}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""
USER_PROMPT_LEAN_WORKBOOK = """I need to solve the following task in Lean 4: \n```\n{informal_statement}\n```\n#####\n\n More formally, I need to prove the following theorem in Lean 4: \n```\n{formal_statement}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""

GOEDEL_AUX_DICT = None # auxiliary dataset for full proofs from Goedel for Lean Workbook, not relevant otherwise
match DATA_SOURCE_ENUM:
    case DataSource.VERINA:
        DATASET = load_dataset("sunblaze-ucb/verina", split="train") # keys: ['id', 'description', 'lean_code', 'signature', 'metadata', 'tests', 'reject_inputs', 'difficulty']
        SPLIT = "test"
    case DataSource.FVAPPS:
        DATASET = load_dataset("quinn-dougherty/fvapps", split="train") # keys: ['apps_id', 'apps_question', 'spec', 'units', 'sorries', 'apps_difficulty', 'assurance_level']
        SPLIT = "rl" # set as train for now, then randomly sample ~200 for validation after
    case DataSource.MINIF2F:
        DATASET = load_dataset("Tonic/MiniF2F", split="train") # keys: ['name', 'split', 'informal_prefix', 'formal_statement', 'goal', 'header']
        SPLIT = "test"
    case DataSource.LEAN_WORKBOOK:
        DATASET = load_dataset("internlm/Lean-Workbook", split="train") # keys: ['id', 'status', 'tactic', 'state_before', 'state_after', 'natural_language_statement', 'answer', 'formal_statement']
        goedel_dataset = load_dataset("Goedel-LM/Lean-workbook-proofs", split="train") # keys: ['problem_id', 'full_proof']
        GOEDEL_AUX_DICT = {item["problem_id"]: item for item in goedel_dataset}
        SPLIT = "rl"
    case _:
        raise ValueError(f"Unsupported data source: {DATA_SOURCE_ENUM}")

BUILD_FVAPPS_INCREMENTAL = False # if True, we build up the dependent proofs/defs incrementally (i.e.  sorry, 2 sorries, 3 sorries, etc.) for FVAPPS

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
    ground_truth = "\n".join(remaining_lines[benchmark_start_idx + 1:]) if all("sorry" not in line for line in remaining_lines) else None

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

def parse_goedel_proof(problem_id):
    full_proof = GOEDEL_AUX_DICT.get(problem_id, {}).get("full_proof", "")
    if not full_proof:
        return {"context": "", "ground_truth": ""}

    # Remove multiline comments
    no_multiline = re.sub(r'/-(.|\n)*?-/', '', full_proof)

    # Remove full-line single-line comments
    cleaned_lines = []
    for line in no_multiline.splitlines():
        if line.strip().startswith("--"):
            continue
        cleaned_lines.append(line)

    cleaned_code = "\n".join(cleaned_lines)

    # Find the last newline before 'theorem'
    theorem_start = cleaned_code.find("theorem")
    if theorem_start == -1:
        raise ValueError("No 'theorem' found in the proof.")
    
    context_end = cleaned_code.rfind("\n", 0, theorem_start) + 1
    context = cleaned_code[:context_end]

    # Find the end of '... := ... by' and first newline after it
    by_match = re.search(r"theorem\s.*?:=\s*.*?by\b", cleaned_code[context_end:], re.DOTALL)
    if not by_match:
        raise ValueError("No 'theorem ... := ... by' found after context.")

    by_end = context_end + by_match.end()
    next_newline = cleaned_code.find("\n", by_end)
    if next_newline == -1:
        ground_truth = ""  # No content after 'by'
    else:
        ground_truth = cleaned_code[next_newline + 1:]  # Everything *after* that newline

    return {
        "context": context,
        "ground_truth": ground_truth
    }

    # print(res["context"])
    # print("#############################################################################################")
    # print(res["ground_truth"])

def parse_fvapps_spec(spec):
    blocks = []
    start = 0

    for match in re.finditer(r'\bsorry\b', spec):
        end = match.end()
        block = spec[start:end]
        blocks.append(block)
        start = end  # next block starts after this "sorry"

    return blocks

def split_train_val(input_jsonl, val_size=200, seed=42):
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    base = os.path.basename(input_jsonl)
    prefix = "lean-rl-data-"
    if not base.startswith(prefix):
        raise ValueError(f"Filename must start with '{prefix}'")
    name = base[len(prefix):]
    train_path = f"lean-train-rl-data-{name}"
    val_path = f"lean-val-rl-data-{name}"

    # Shuffle and sample
    random.seed(seed)
    val_indices = set(random.sample(range(len(lines)), val_size))

    # Write to output files
    with open(train_path, 'w', encoding='utf-8') as train_f, \
         open(val_path, 'w', encoding='utf-8') as val_f:
        for i, line in enumerate(lines):
            if i in val_indices:
                val_f.write(line)
            else:
                train_f.write(line)
    print(f"Saved {len(lines) - val_size} entries to {train_path} and {val_size} entries to {val_path}")

def save_entry(datum_id, user_prompt, ground_truth, context, formal_statement):
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
            "context": context, # for debugging reconstructing full Lean executable
            "formal_statement": formal_statement, # for debugging reconstructing full Lean executable
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "tools/execute_lean": {
                    "example_name": datum_id,                }
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

def augment():
    print(f"Starting RL augmentation from index {START_INDEX} for {NUM_EXAMPLES} examples in data source {DATA_SOURCE}.")
    augmented_count = 0

    for datum in TRAIN_DATA[START_INDEX:START_INDEX + NUM_EXAMPLES]:
        match DATA_SOURCE_ENUM:
            case DataSource.VERINA: # keys: ['id', 'description', 'lean_code', 'signature', 'metadata', 'tests', 'reject_inputs', 'difficulty']
                parsed_lean_code = parse_verina_lean_code(datum['lean_code'])
                context, formal_statement = parsed_lean_code["context"], parsed_lean_code["formal_statement"]
                user_prompt = USER_PROMPT_VERINA.format(
                    informal_statement=datum["description"], 
                    formal_statement=formal_statement, 
                    file_context=context
                )
                datum_id = datum["id"]
                ground_truth = parsed_lean_code["ground_truth"]
            case DataSource.FVAPPS: # keys: ['apps_id', 'apps_question', 'spec', 'units', 'sorries', 'apps_difficulty', 'assurance_level']
                if BUILD_FVAPPS_INCREMENTAL:
                    specs = parse_fvapps_spec(datum["spec"]) # build up the dependent proofs/defs incrementally (i.e.  sorry, 2 sorries, 3 sorries, etc.)
                    subproofs_info = {}
                    for i, spec in enumerate(specs, start=1):
                        user_prompt = USER_PROMPT_FVAPPS.format(
                            question=datum["apps_question"], 
                            num_sorries=datum['sorries'], 
                            spec=spec
                        )
                        context = "" # in FVAPPS, we replace the sorries directly with proofs in the file, so we don't record context here
                        datum_id = datum["apps_id"] + f"_{i}" 
                        ground_truth = None
                        subproofs_info[datum_id] = {
                            "user_prompt": user_prompt,
                            "ground_truth": None,
                            "context": context
                        }
                else: # just use the full spec with all sorries
                    user_prompt = USER_PROMPT_FVAPPS.format(
                        question=datum["apps_question"], 
                        num_sorries=datum['sorries'], 
                        spec=datum["spec"]
                    )
                    context, formal_statement = datum["spec"], "" # in FVAPPS, we replace the sorries directly with proofs in the spec
                    datum_id = datum["apps_id"]
                    ground_truth = None 
            case DataSource.MINIF2F: # keys: ['name', 'split', 'informal_prefix', 'formal_statement', 'goal', 'header']
                context, formal_statement = datum["header"], datum["formal_statement"]
                user_prompt = USER_PROMPT_MINIF2F.format(
                    informal_statement=datum["informal_prefix"], 
                    formal_statement=datum["formal_statement"], 
                    file_context=context
                )
                datum_id = datum["name"]
                ground_truth = None
            case DataSource.LEAN_WORKBOOK: # keys: ['id', 'status', 'tactic', 'state_before', 'state_after', 'natural_language_statement', 'answer', 'formal_statement']
                parsed_goedel_proof = parse_goedel_proof(datum['id'])
                context, ground_truth = parsed_goedel_proof["context"], parsed_goedel_proof["ground_truth"]
                if not ground_truth:
                    print(f"Skipping Lean Workbook example {datum['id']} due to missing ground truth and uncertain provability.")
                    continue
                formal_statement = datum["formal_statement"]
                if formal_statement.endswith(" sorry"):
                    formal_statement = formal_statement[:-6]  # remove the trailing " sorry"
                user_prompt = USER_PROMPT_LEAN_WORKBOOK.format(
                    informal_statement=datum["natural_language_statement"], 
                    formal_statement=formal_statement,
                    file_context=context
                )
                datum_id = datum["id"]
            case _:
                raise ValueError(f"Unsupported data source: {DATA_SOURCE_ENUM}")

        if DATA_SOURCE_ENUM == DataSource.FVAPPS and BUILD_FVAPPS_INCREMENTAL: # save all the subproofs we build incrementally from the spec
            for datum_id, info in subproofs_info.items():
                save_entry(datum_id, info["user_prompt"], info["ground_truth"], info["context"])
                augmented_count += 1
                print(f"{augmented_count+START_INDEX}/{NUM_EXAMPLES} done  ->  {SAVE_LOC}", file=sys.stderr)
        else:
            save_entry(datum_id, user_prompt, ground_truth, context, formal_statement)
            augmented_count += 1
            print(f"{augmented_count+START_INDEX}/{NUM_EXAMPLES} done  ->  {SAVE_LOC}", file=sys.stderr)

    print(f"Wrote {augmented_count} files to “{SAVE_LOC}”.")
    
if __name__ == "__main__":
    augment()
    combine_to_jsonl(SAVE_LOC, OUT_PATH_JSONL)
    if SPLIT == "rl":
        split_train_val(OUT_PATH_JSONL)