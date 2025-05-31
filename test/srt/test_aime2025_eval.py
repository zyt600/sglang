import json
import os
import unittest
import warnings
from datetime import datetime
from types import SimpleNamespace
from datasets import load_dataset

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

MODEL_SCORE_THRESHOLDS = {
    "Qwen/QwQ-32B": 0.4, # in practice, 0.533
    "XiaomiMiMo/MiMo-7B-SFT": 0.4,
}

DEFAULT_MODELS = [
    "Qwen/QwQ-32B",
    "XiaomiMiMo/MiMo-7B-SFT",
]

DEFAULT_SCORE_TYPE = "pass@1"  # pass@1, pass@k, mean
DEFAULT_K = 5
DEFAULT_NUM_SAMPLES = 5
DEFAULT_MAX_EXAMPLES = 20
MAX_TOKEN = 30000


def parse_models(model_string):
    return [model.strip() for model in model_string.split(",") if model.strip()]


def load_aime2025():
    ds = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    return ds


def get_answer(example):
    print("get_answer called with example:", example)
    print()
    print()
    return str(example["answer"]).strip()


def extract_answer_from_output(output):
    import re
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(boxed_pattern, output)
    if matches:
        return matches[-1].strip()
    
    number_pattern = r"(\d+)"
    numbers = re.findall(number_pattern, output)
    if numbers:
        return numbers[-1]
    
    return ""


def score_pass_at_k(preds, gold, k):
    for pred in preds[:k]:
        if pred.strip() == gold:
            return 1
    return 0


def score_mean(preds, gold):
    return sum([1 if pred.strip() == gold else 0 for pred in preds]) / len(preds)


def popen_launch_server_wrapper(base_url, model, tp_size=2):
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]
    if tp_size > 1:
        other_args.extend(["--tp", str(tp_size)])
    
    other_args.extend(["--download-dir", "/home/ytzhou/.cache/huggingface/hub"])

    print("before popen_launch_server")
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    print("after popen_launch_server")
    return process


def evaluate_model_with_sglang(base_url, model_name, ds, score_type, k, num_samples, max_examples):
    print(f"\nEvaluating model {model_name} ...")
    sgl.set_default_backend(sgl.RuntimeEndpoint(base_url))
    results = []
    correct = 0
    total = 0

    @sgl.function
    def solve_aime_problem(s, question):
        s += "You are a helpful AI assistant that solves math problems step by step.\n"
        s += "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        s += f"Problem: {question}\n"
        s += "Solution: " + sgl.gen("answer", max_tokens=MAX_TOKEN)

    for i, example in enumerate(ds):
        if i >= max_examples:
            break
        question = example["question"]
        gold = get_answer(example)
        preds = []
        try:
            state = solve_aime_problem.run(question=question, temperature=0.7)
            print("state:", state)
            if "answer" in state:
                output = state["answer"]
                pred = extract_answer_from_output(output)
            else:
                print(f"Warning: No answer generated for question {i+1}")
                pred = ""
            preds = [pred]
            score = 1 if pred == gold else 0
            results.append({
                "question": question,
                "gold": gold,
                "preds": preds,
                "score": score
            })
            correct += score
            total += 1
            if (i + 1) % 5 == 0:
                print(f"Evaluated {i+1} questions, current accuracy: {correct/total:.3f}")
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            score = 0.0
        results[-1]["score"] = score

    final_score = correct / total if total > 0 else 0.0
    print(f"Model {model_name} final score: {final_score:.4f}")
    return {
        "score": final_score,
        "correct": correct,
        "total": total,
        "details": results
    }


def write_results_to_json(model, metrics, mode="a"):
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "metrics": metrics,
        "score": metrics["score"],
    }

    existing_results = []
    if mode == "a" and os.path.exists("results.json"):
        try:
            with open("results.json", "r") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    if isinstance(existing_results, list):
        existing_results.append(result)
    else:
        existing_results = [result]

    with open("results.json", "w") as f:
        json.dump(existing_results, f, indent=2)


def check_model_scores(results):
    failed_models = []
    summary = " | model | score | threshold |\n"
    summary += "| ----- | ----- | --------- |\n"

    for model, score in results:
        threshold = MODEL_SCORE_THRESHOLDS.get(model)
        if threshold is None:
            print(f"Warning: No threshold defined for model {model}")
            continue

        if score < threshold:
            failed_models.append(
                f"\nScore Check Failed: {model}\n"
                f"Model {model} score ({score:.4f}) is below threshold ({threshold:.4f})"
            )

        line = f"| {model} | {score} | {threshold} |\n"
        summary += line

    print(summary)

    if is_in_ci():
        write_github_step_summary(f"### TestAIME2025Eval\n{summary}")

    if failed_models:
        raise AssertionError("\n".join(failed_models))


class TestAIME2025Eval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = parse_models(",".join(DEFAULT_MODELS))
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.ds = load_aime2025()
        cls.score_type = DEFAULT_SCORE_TYPE
        cls.k = DEFAULT_K
        cls.num_samples = DEFAULT_NUM_SAMPLES
        cls.max_examples = DEFAULT_MAX_EXAMPLES

    def test_aime2025_all_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []

        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model)
                
                try:
                    print("before evaluate_model_with_sglang")
                    metrics = evaluate_model_with_sglang(
                        self.base_url,
                        model,
                        self.ds,
                        self.score_type,
                        self.k,
                        self.num_samples,
                        self.max_examples
                    )
                    print("after evaluate_model_with_sglang")
                    print(
                        f"{'=' * 42}\n{model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )

                    write_results_to_json(model, metrics, "w" if is_first else "a")
                    is_first = False

                    all_results.append((model, metrics["score"]))
                    
                finally:
                    kill_process_tree(process.pid)

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results.json: {e}")

        check_model_scores(all_results)


if __name__ == "__main__":
    unittest.main() 