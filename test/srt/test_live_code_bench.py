import json
import os
import unittest
import warnings
from datetime import datetime
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
    "Qwen/QwQ-32B": 0.4,
    "XiaomiMiMo/MiMo-7B-SFT": 0.4,
}

DEFAULT_MODELS = [
    "Qwen/QwQ-32B",
    "XiaomiMiMo/MiMo-7B-SFT",
]

DEFAULT_SCORE_TYPE = "pass@1"
DEFAULT_K = 5
DEFAULT_NUM_SAMPLES = 5
DEFAULT_MAX_EXAMPLES = 20
MAX_TOKEN = 2048


def parse_models(model_string):
    return [model.strip() for model in model_string.split(",") if model.strip()]


def load_livecodebench():
    ds = load_dataset("ise-uiuc/LiveCodeBench", split="test")
    return ds


def get_answer(example):
    return str(example["reference_solution"]).strip()


def extract_answer_from_output(output):
    return output.strip()


def score_pass_at_k(preds, gold, k):
    for pred in preds[:k]:
        if pred.strip() == gold:
            return 1
    return 0


def popen_launch_server_wrapper(base_url, model, tp_size=2):
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]
    if tp_size > 1:
        other_args.extend(["--tp", str(tp_size)])

    other_args.extend(["--download-dir", "/home/ytzhou/.cache/huggingface/hub"])

    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


def evaluate_model_with_sglang(base_url, model_name, ds, score_type, k, num_samples, max_examples):
    print(f"\nEvaluating model {model_name} ...")
    sgl.set_default_backend(sgl.RuntimeEndpoint(base_url))
    results = []
    correct = 0
    total = 0

    @sgl.function
    def solve_code_task(s, prompt):
        s += "Write a correct and complete Python function for the following task:\n"
        s += f"{prompt}\n"
        s += sgl.gen("answer", max_tokens=MAX_TOKEN)

    for i, example in enumerate(ds):
        if i >= max_examples:
            break
        prompt = example["prompt"]
        gold = get_answer(example)
        preds = []
        try:
            state = solve_code_task.run(prompt=prompt, temperature=0.2)
            if "answer" in state:
                output = state["answer"]
                pred = extract_answer_from_output(output)
            else:
                print(f"Warning: No answer generated for task {i+1}")
                pred = ""
            preds = [pred]
            score = 1 if pred == gold else 0
            results.append({
                "prompt": prompt,
                "gold": gold,
                "preds": preds,
                "score": score
            })
            correct += score
            total += 1
            if (i + 1) % 5 == 0:
                print(f"Evaluated {i+1} tasks, current accuracy: {correct/total:.3f}")
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            results.append({"prompt": prompt, "error": str(e)})
            continue

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
    if mode == "a" and os.path.exists("results_livecodebench.json"):
        try:
            with open("results_livecodebench.json", "r") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    if isinstance(existing_results, list):
        existing_results.append(result)
    else:
        existing_results = [result]

    with open("results_livecodebench.json", "w") as f:
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
        write_github_step_summary(f"### TestLiveCodeBenchEval\n{summary}")

    if failed_models:
        raise AssertionError("\n".join(failed_models))


class TestLiveCodeBenchEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = parse_models(",".join(DEFAULT_MODELS))
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.ds = load_livecodebench()
        cls.score_type = DEFAULT_SCORE_TYPE
        cls.k = DEFAULT_K
        cls.num_samples = DEFAULT_NUM_SAMPLES
        cls.max_examples = DEFAULT_MAX_EXAMPLES

    def test_livecodebench_all_models(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first = True
        all_results = []

        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model)

                try:
                    metrics = evaluate_model_with_sglang(
                        self.base_url,
                        model,
                        self.ds,
                        self.score_type,
                        self.k,
                        self.num_samples,
                        self.max_examples
                    )

                    print(
                        f"{'=' * 42}\n{model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
                    )

                    write_results_to_json(model, metrics, "w" if is_first else "a")
                    is_first = False

                    all_results.append((model, metrics["score"]))

                finally:
                    kill_process_tree(process.pid)

        try:
            with open("results_livecodebench.json", "r") as f:
                print("\nFinal Results from results_livecodebench.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results_livecodebench.json: {e}")

        check_model_scores(all_results)


if __name__ == "__main__":
    unittest.main()
