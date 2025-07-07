import json, sys, time, subprocess
from pathlib import Path

from gen_sft import combine_to_jsonl
from datasets import load_dataset

USER_PROMPT_FORMAT_VERINA = """The theorem I'm trying to prove is \n```\n{theorem}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""
USER_PROMPT_FORMAT_FVAPPS = """The question I need to answer in Lean 4 is as follows: \n```\n{question}\n```\n#####\n\n. To answer this question, you need to replace the {num_sorries} \"sorry\" keywords in the following Lean spec with Lean proofs that make the resulting Lean executable work: \n```\n{spec}\n```\n#####\n\n. Your answer should satisfy the following unit tests: \n```\n{unit_tests}\n```\n#####\n\n. I need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""
USER_PROMPT_MINIF2F = """I need to solve the following task in Lean: {informal_statement}. More formally, theorem I'm trying to prove is \n```\n{theorem}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\nI need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""

SPLIT = "test"
DATA_SOURCE = "MiniF2F" 

SAVE_LOC = Path(f"lean-{SPLIT}-data-{DATA_SOURCE}")
SAVE_LOC.mkdir(parents=True, exist_ok=True)
OUT_PATH_JSONL = SAVE_LOC.with_suffix(".jsonl")

# DATASET = load_dataset("quinn-dougherty/fvapps", split="train") # keys: ['apps_id', 'apps_question', 'spec', 'units', 'sorries', 'apps_difficulty', 'assurance_level']
# DATASET = load_dataset("sunblaze-ucb/verina", split="train") # keys: ['id', 'description', 'lean_code', 'signature', 'metadata', 'tests', 'reject_inputs', 'difficulty']
DATASET = load_dataset("Tonic/MiniF2F", split="train") # keys: ['name', 'split', 'informal_prefix', 'formal_statement', 'goal', 'header']
TRAIN_DATA = list(DATASET) # this should load in order, can write sorting function if we run into issues

NUM_EXAMPLES = len(TRAIN_DATA)
START_INDEX = sum(1 for f in SAVE_LOC.iterdir() if f.is_file())

def augment():
    print(f"Starting RL augmentation from index {START_INDEX} for {NUM_EXAMPLES} examples in data source {DATA_SOURCE}.")
    augmented_count = 0

    for datum in TRAIN_DATA[START_INDEX:START_INDEX + NUM_EXAMPLES]:
        # user_prompt = USER_PROMPT_FORMAT_FVAPPS.format(question=datum["apps_question"], num_sorries=datum['sorries'], spec=datum["spec"], unit_tests=datum["units"])
        # user_prompt = USER_PROMPT_FORMAT_VERINA.format(question=datum["apps_question"], num_sorries=datum['sorries'], spec=datum["spec"], unit_tests=datum["units"])
        user_prompt = USER_PROMPT_MINIF2F.format(informal_statement=datum["informal_prefix"], theorem=datum["formal_statement"], file_context=datum["header"])
        
        entry = {
            "data_source": DATA_SOURCE,
            "user_prompt": user_prompt,
            "name":  datum["name"],
        }

        # ──────────────────────────────────────────────────────────────
        # Write ONE file per example
        #   e.g.  sft-data/train-sft_226.json
        # ──────────────────────────────────────────────────────────────
        out_path = SAVE_LOC / f"{datum['name']}.json"
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(entry, f_out, ensure_ascii=False, indent=2)

        augmented_count += 1
        print(f"{augmented_count+START_INDEX}/{NUM_EXAMPLES} done  ->  {SAVE_LOC}", file=sys.stderr)

    print(f"Wrote {augmented_count} files to “{SAVE_LOC}”.")
    
if __name__ == "__main__":
    # augment()
    combine_to_jsonl(SAVE_LOC, OUT_PATH_JSONL)