import json, sys, time

from trapi_DO_NOT_COMMIT import trapi_call
from pathlib import Path
from azure.core.exceptions import ServiceResponseError, HttpResponseError
from datasets import load_dataset

def get_prompt_reasoning(language, instruction, response): 
    # lean_workbook_plus_10224 in Lean Workbook / Goedel dataset
    return f"""You are a synthetic data augmentator that generates high-quality instruction data for training a LLM that is fluent in {language}.

    Given an instruction and a final code, your task is to generate a chain-of-thought reasoning that an expert might used to generate the code from the instructions.
    1. You are NOT simply explaining how the code works, but rather how do you derive the code from the natural language instruction.
    2. Include great details how to implement the instructions' requirements in {language} BEFORE the actual {language} code snippet.
    This is to make sure that the model trained on this data can properly learn how to condition on the natural language explanations to generate the correct snippet.
        - When writing a lemma or theorem (using `Lemma` or `Theorem`), you should first:
        * Examine the type signature of the lemma or theorem.
        * Identify the main goal of the lemma or theorem.
        * Describe the mathematical or logical properties that the lemma or theorem is asserting.
        * Discuss any assumptions or hypotheses that are necessary for the proof.
        * Outline the proof strategy, including the tactics you plan to use.
        Only after this should you write the proof script.

    - When writing a proposition or predicate (e.g., `Definition` with a `Prop` type), you should first:
        * Formalize the properties being checked.
        * Specify the type of the proposition.
        * State any type restrictions or constraints explicitly (using dependent types or curly braces for arguments).
        Then provide the proposition definition.

    - When writing a function (using `Definition` or `Fixpoint` returning `Set`, `Type`, or `bool`, etc.), you should first:
        * Formalize the function's type signature.
        * Describe the high-level functionality.
        * Provide the detailed algorithmic implementation.
        * Explain how the implementation meets the specification.
        After this explanation, include the actual function code.

    3. Your explanations should be very very detailed like explaining to a beginner, in spoken language style, and do not use bullet lists and hierarchical markdowns. Do not care about the length of your output, you are obliged to produce the most detailed output you can. Just make sure it is very detailed and thorough, as shown in the examples. Specifically, for each code snippet (theorem, function, proposition/predicate), you MUST write at least 5 sentences of description before each code snippet.

    4. However, your explanations should not be too verbose, and should not include any unnecessary information. Do not repeat the same information multiple times, and do not include any irrelevant information. The explanations should be concise and to the point, while still being detailed enough to understand the code. Try to keep your reasoning under approximately {len(response)} words.
    
    5. Remembers, ALL of the details about how to implement the instructions' requirements in {language} need to come BEFORE the actual {language} code snippet. There should be no additional explanations after the code snippet, except for the final summary of the proof or function.

    6. Importantly, ALL code should be in Lean 4. There should be NO code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features.

    Instruction example: Check whether the inequality 1 + x^2 + y^2 + 2xy <= (4/3) * (1 + x^2) * (1 + y^2) holds for the given real numbers x and y.

    Final code example:

    import Mathlib
    import Aesop

    set_option maxHeartbeats 0

    open BigOperators Real Nat Topology Rat

    theorem lean_workbook_plus_10224 (x y : \mathbb{{R}}) : 1 + x^2 + y^2 + 2 * x * y <= (4:\mathbb{{R}}) / 3 * (1 + x^2) * (1 + y^2) := by
    nlinarith [sq_nonneg (x - y), sq_nonneg (x + y), sq_nonneg (2 * x * y - 1), sq_nonneg (2 * x * y + 1),
    sq_nonneg (x * y - 1), sq_nonneg (x * y + 1), sq_nonneg (x - y + 1), sq_nonneg (x + y - 1)]

    Chain-of-thought example:
    
    Let's think step-by-step. We are asked to prove the inequality:
    1 + x^2 + y^2 + 2xy <= (4/3) * (1 + x^2) * (1 + y^2)

    Step 1: Context from Cauchy-Schwarz

    By the Cauchy-Schwarz inequality, for vectors u and v,
        (∑ u_i v_i)^2 <= (∑ u_i^2)(∑ v_i^2)
    In the special case where u = (1, x) and v = (1, y), we have:
        ∑ u_i v_i = 1 + x*y
        ∑ u_i^2 = 1 + x^2
        ∑ v_i^2 = 1 + y^2
    So,
        (1 + x*y)^2 <= (1 + x^2)(1 + y^2)
    Expanding the left-hand side:
        1 + 2xy + x^2 y^2
    Expanding the right-hand side:
        1 + x^2 + y^2 + x^2 y^2

    So this tells us:
        1 + 2xy + x^2 y^2 <= 1 + x^2 + y^2 + x^2 y^2
    Subtracting 1 + x^2 y^2 from both sides:
        2xy <= x^2 + y^2

    Now, adding 1 + x^2 + y^2 to both sides:
        1 + x^2 + y^2 + 2xy <= 1 + 2x^2 + 2y^2

    That gives us a rough bound, but the target inequality is stronger:
        1 + x^2 + y^2 + 2xy <= (4/3)(1 + x^2)(1 + y^2)

    Step 2: Strategy

    Since we're working over the reals, and the expression involves squares and products of x and y,
    we can try to prove this using nonlinear arithmetic and the fact that squares are non-negative.

    Step 3: Applying Nonlinear Arithmetic in Lean
    
    We invoke the `nlinarith` tactic, which can solve nonlinear inequalities involving real numbers
    by reasoning about non-negativity of expressions and reducing to linear inequalities.

    To help `nlinarith`, we provide auxiliary facts: several square expressions which are known to be non-negative.
    These facts include:
        (x - y)^2 >= 0
        (x + y)^2 >= 0
        (2xy - 1)^2 >= 0
        (2xy + 1)^2 >= 0
        (xy - 1)^2 >= 0
        (xy + 1)^2 >= 0
        (x - y + 1)^2 >= 0
        (x + y - 1)^2 >= 0

    Each of these is of the form `sq_nonneg(expression)`, asserting that the square of a real-valued
    expression is non-negative.
c
    These inequalities enrich the searh space for `nlinarith` and allow it to verify the desired inequality.

    Step 4: Conclusion
    
    The tactic `nlinarith` successfully concludes the proof using these non-negative square expressions.

    Thus, we have proven that:
        1 + x^2 + y^2 + 2xy <= (4/3)(1 + x^2)(1 + y^2)
    as desired.

    Instruction:
    {instruction}

    Final code:
    {response}

    Chain-of-thought:
    """

MAX_RETRIES = 3          # how many times to retry the call
BACKOFF     = 5          # seconds to wait between retries
USER_PROMPT_FORMAT = """The theorem I'm trying to prove is \n```\n{theorem}\n```\n#####\n\nThe file context in which I'm writing the proof is \n```\n{file_context}\n```\n#####\n\n. I need ALL code to be in Lean 4. I cannot have ANY code written in Lean 3 whatsoever. DO NOT use Lean 3 syntax or features."""

SAVE_LOC = Path("lean-sft-data-deepseekV1")
SAVE_LOC.mkdir(parents=True, exist_ok=True)
OUT_PATH_JSONL = SAVE_LOC.with_suffix(".jsonl")

SPLIT = "train-sft"
NUM_EXAMPLES = 27550 
START_INDEX = sum(1 for f in SAVE_LOC.iterdir() if f.is_file())

def combine_to_jsonl(src_dir: Path, out_file: Path) -> None:
    """
    Read every *.json file in `src_dir` and append its (single) JSON object
    as one line to `out_file` in JSON‑Lines format.
    Existing output is overwritten.
    """
    with out_file.open("w", encoding="utf-8") as fout:
        # sort() gives stable order; drop if you don’t care
        for fp in sorted(src_dir.glob("*.json")):
            with fp.open("r", encoding="utf-8") as fin:
                obj = json.load(fin)          # each file contains ONE object
            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")                  # newline separates entries
    print(f"Wrote {out_file} with {len(list(src_dir.glob('*.json')))} lines.")

def safe_trapi_call(prompt: str):
    """
    Wrapper around trapi_call that retries a few times and returns None
    if it still fails.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return trapi_call(prompt)
        except (ServiceResponseError, HttpResponseError) as e:
            print(f"[trapi_call] attempt {attempt}/{MAX_RETRIES} failed: {e}",
                  file=sys.stderr)
            if attempt == MAX_RETRIES:
                # give up after last attempt
                return None
            time.sleep(BACKOFF)      # simple linear back-off (or use 2**attempt)
        except Exception as e:
            # Catch-all so one unexpected error doesn't crash you
            print(f"[trapi_call] unexpected error: {e}", file=sys.stderr)
            return None

def augment():
    print(f"Starting SFT augmentation from index {START_INDEX} for {NUM_EXAMPLES} examples.")
    augmented_count = 0

    dataset = load_dataset("deepseek-ai/DeepSeek-Prover-V1") # keys: ['name', 'split', 'formal_statement', 'goal', 'header', 'formal_proof']
    train_data = list(dataset["train"]) # this should load in order, can write sorting function if we run into issues

    for datum in train_data[START_INDEX:START_INDEX + NUM_EXAMPLES]:
        user_prompt = USER_PROMPT_FORMAT.format(theorem=datum["formal_statement"], file_context=datum["header"])
        proof       = datum["formal_proof"]
        prompt      = get_prompt_reasoning(language="Coq",
                                    instruction=user_prompt,
                                    response=proof)

        response  = safe_trapi_call(prompt)
        if not response:
            print(f"[augment] Failed to get response for theorem {datum['name']}. Skipping.", file=sys.stderr)
            continue

        reasoning = response.strip()
        if not reasoning:
            print(f"[augment] Empty reasoning for theorem {datum['name']}. Skipping.", file=sys.stderr)
            continue

        entry = {
            "data_source": "DeepSeek-Prover-V1",
            "prompt": [{"role": "user", "content": user_prompt}],
            "ability": "coding/Coq",
            "reward_model": {"style": "execution"},
            "extra_info": {
                "question":          user_prompt,
                "reasoning":         reasoning,
                "answer":            proof,
                "formatted_response": (
                    f"<think>\n{reasoning}\n</think>\n"
                    f"<answer>\n{proof}\n</answer>"
                ),
                "name":  datum["name"], # THIS IS THE ID FOR THIS DATASET
            },
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
    augment()
    # combine_to_jsonl(SAVE_LOC, OUT_PATH_JSONL)