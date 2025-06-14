import json
import os
import unittest
import warnings
from datetime import datetime
from types import SimpleNamespace
from datasets import load_dataset
from PIL import Image

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
    "Qwen/Qwen2.5-VL-32B-Instruct": 0.49,
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ": 0.50,
    "OpenGVLab/InternVL3-78B-AWQ": 0.72,
    "OpenGVLab/InternVL3-38B": 0.70,
}

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    "OpenGVLab/InternVL3-78B-AWQ",
    "OpenGVLab/InternVL3-38B",
]

DEFAULT_SCORE_TYPE = "pass@1"
DEFAULT_K = 1
DEFAULT_NUM_SAMPLES = 1
DEFAULT_MAX_EXAMPLES = 10
MAX_TOKENS_OUTPUT = 128

# ['standard (10 options)', 'standard (4 options)', 'vision']
DEFAULT_MMMU_CONFIG = 'standard (4 options)'


def parse_models(model_string):
    return [model.strip() for model in model_string.split(",") if model.strip()]


def load_mmmu_pro_dataset():
    print(f"Loading 'mmmu/mmmu_pro' dataset with config: '{DEFAULT_MMMU_CONFIG}' and split: 'test'")
    try:
        ds = load_dataset("mmmu/mmmu_pro", name=DEFAULT_MMMU_CONFIG, split="test")
    except Exception as e:
        print(f"Failed to load 'mmmu/mmmu_pro' dataset: {e}")
        print("Please check if the dataset name and configuration name are correct.")
        raise
    return ds


def get_answer_mmmu(example):
    return str(example["answer"]).strip()


def extract_answer_from_output_mmmu(output: str):
    import re
    matches = re.findall(r'[A-Z]', output)
    if matches:
        return matches[-1]
    return ""


def popen_launch_server_wrapper(base_url, model, tp_size=1):
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]
    if tp_size > 1:
        other_args.extend(["--tp", str(tp_size)])
    
    print(f"Launching sglang server for model {model}...")
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
        other_args=other_args,
    )
    print(f"sglang server for model {model} launched (PID: {process.pid}).")
    return process


def evaluate_model_with_sglang_mmmu(base_url, model_name, ds, score_type, k, num_samples, max_examples):
    print(f"\nEvaluating model {model_name} on MMMU-Pro (config: {DEFAULT_MMMU_CONFIG})...")
    sgl.set_default_backend(sgl.RuntimeEndpoint(base_url))
    results_details = []
    correct_predictions = 0
    total_evaluated = 0

    @sgl.function
    def solve_mmmu_problem(s, image_data, question_text, options_list):
        s += s.image(image_data)
        s += f"\nQuestion: {question_text}\n"
        s += "Options:\n"
        for i, opt_text in enumerate(options_list):
            s += f"{chr(65+i)}. {opt_text}\n"
        s += "Please choose the correct option and provide only the letter of your choice (e.g., A, B, C, D).\nAnswer: "
        s += sgl.gen("answer", max_tokens=MAX_TOKENS_OUTPUT)

    num_total_examples = len(ds)
    effective_max_examples = num_total_examples
    if max_examples is not None:
        effective_max_examples = min(num_total_examples, max_examples)

    for i, example in enumerate(ds):
        if i >= effective_max_examples:
            break

        pil_image = example["image"]
        question = example["question"]
        options = example["options"]
        if not isinstance(options, list):
            print(f"Warning: Example {i} has options in unexpected format: {type(options)}. Skipping.")
            continue

        gold_answer = get_answer_mmmu(example)
        
        if not isinstance(pil_image, Image.Image):
            print(f"Warning: Example {i} has image in unexpected format: {type(pil_image)}. Trying to convert.")
            try:
                if hasattr(pil_image, 'convert'):
                     pil_image = pil_image.convert("RGB")
                else:
                    print("Cannot convert image. Skipping.")
                    continue
            except Exception as img_e:
                print(f"Error converting image for example {i}: {img_e}. Skipping.")
                continue
        
        try:
            state = solve_mmmu_problem.run(
                image_data=pil_image,
                question_text=question,
                options_list=options,
                temperature=0.0,
            )

            if "answer" in state:
                model_output = state["answer"]
                predicted_answer = extract_answer_from_output_mmmu(model_output)
            else:
                print(f"Warning: No 'answer' key in state for question {i+1}. Output: {state.text()}")
                predicted_answer = ""
                model_output = state.text()

            is_correct = 1 if predicted_answer == gold_answer else 0
            
            if is_correct:
                correct_predictions += 1
            
            results_details.append({
                "id": example.get("id", str(i)),
                "question": question,
                "options": options,
                "gold": gold_answer,
                "raw_output": model_output,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })
            total_evaluated += 1

            if (i + 1) % 5 == 0:
                current_accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0
                print(f"Evaluated {i+1}/{effective_max_examples} examples. Current Accuracy: {current_accuracy:.3f}")

        except Exception as e:
            print(f"Error evaluating model {model_name} on example {i+1} (ID: {example.get('id', 'N/A')}): {e}")
            total_evaluated += 1

    final_score = correct_predictions / total_evaluated if total_evaluated > 0 else 0.0
    print(f"Model {model_name} final score on MMMU-Pro: {final_score:.4f} ({correct_predictions}/{total_evaluated})")
    return {
        "score": final_score,
        "correct": correct_predictions,
        "total": total_evaluated,
        "details": results_details
    }


def write_results_to_json(model, metrics, filename="results_mmmu_pro.json", mode="a"):
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "dataset_config": DEFAULT_MMMU_CONFIG,
        "metrics": {
            "score": metrics["score"],
            "correct": metrics["correct"],
            "total": metrics["total"],
        },
    }

    all_results_data = []
    if mode == "a" and os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                all_results_data = json.load(f)
        except json.JSONDecodeError:
            all_results_data = []

    if not isinstance(all_results_data, list):
        all_results_data = []

    all_results_data.append(result_entry)

    with open(filename, "w") as f:
        json.dump(all_results_data, f, indent=2)
    
    detailed_filename = f"results_mmmu_pro_{model.replace('/', '_')}_details.json"
    with open(detailed_filename, "w") as f_detail:
        json.dump(metrics["details"], f_detail, indent=2)
    print(f"Detailed results saved to {detailed_filename}")


def check_model_scores(results_summary, thresholds):
    failed_models_messages = []
    summary_table = " | model | score | threshold |\n"
    summary_table += "| ----- | ----- | --------- |\n"

    for model_name, achieved_score in results_summary:
        threshold = thresholds.get(model_name)
        if threshold is None:
            print(f"Warning: No threshold defined for model {model_name}. Skipping check.")
            summary_table += f"| {model_name} | {achieved_score:.4f} | N/A |\n"
            continue

        pass_status = "✅" if achieved_score >= threshold else "❌"
        summary_table += f"| {model_name} | {achieved_score:.4f} | {threshold:.4f} | {pass_status}\n"

        if achieved_score < threshold:
            failed_models_messages.append(
                f"\nScore Check Failed for MMMU-Pro: {model_name}\n"
                f"Model {model_name} score ({achieved_score:.4f}) is below threshold ({threshold:.4f})"
            )
    
    print("\n--- MMMU-Pro Score Summary ---")
    print(summary_table)

    if is_in_ci():
        write_github_step_summary(f"### TestMMMUProEval Summary\n{summary_table}")

    if failed_models_messages:
        raise AssertionError("\n".join(failed_models_messages))


class TestMMMUProEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_to_test = parse_models(",".join(DEFAULT_MODELS))
        cls.base_url = DEFAULT_URL_FOR_TEST
        print("Loading MMMU-Pro dataset...")
        cls.dataset = load_mmmu_pro_dataset()
        print(f"MMMU-Pro dataset loaded. Number of examples: {len(cls.dataset)}")
        cls.score_type = DEFAULT_SCORE_TYPE
        cls.k_value = DEFAULT_K
        cls.num_samples_per_problem = DEFAULT_NUM_SAMPLES
        cls.max_examples_to_run = DEFAULT_MAX_EXAMPLES

    def test_mmmu_pro_evaluation(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        is_first_model = True
        all_model_scores = []

        for model_name in self.models_to_test:
            with self.subTest(model=model_name):
                sglang_server_process = None
                try:
                    sglang_server_process = popen_launch_server_wrapper(self.base_url, model_name)
                    
                    print(f"Starting evaluation for model: {model_name}")
                    evaluation_metrics = evaluate_model_with_sglang_mmmu(
                        self.base_url,
                        model_name,
                        self.dataset,
                        self.score_type,
                        self.k_value,
                        self.num_samples_per_problem,
                        self.max_examples_to_run
                    )
                    print(f"Evaluation finished for model: {model_name}")
                    print(
                        f"{'=' * 42}\n"
                        f"Model: {model_name} on MMMU-Pro\n"
                        f"Metrics: score={evaluation_metrics['score']:.4f}, "
                        f"correct={evaluation_metrics['correct']}, "
                        f"total={evaluation_metrics['total']}\n"
                        f"{'=' * 42}\n"
                    )

                    write_results_to_json(
                        model_name, 
                        evaluation_metrics, 
                        mode="w" if is_first_model else "a"
                    )
                    is_first_model = False
                    all_model_scores.append((model_name, evaluation_metrics["score"]))
                    
                except Exception as e:
                    self.fail(f"Evaluation failed for {model_name} due to: {e}")
                finally:
                    if sglang_server_process:
                        print(f"Shutting down sglang server for {model_name} (PID: {sglang_server_process.pid})...")
                        kill_process_tree(sglang_server_process.pid)
                        sglang_server_process.wait()
                        print(f"Server for {model_name} shut down.")

        results_filename = "results_mmmu_pro.json"
        try:
            if os.path.exists(results_filename):
                with open(results_filename, "r") as f:
                    print(f"\nFinal aggregated results from {results_filename}:")
                    print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading {results_filename}: {e}")

        if all_model_scores:
             check_model_scores(all_model_scores, MODEL_SCORE_THRESHOLDS)
        else:
            print("No models were successfully evaluated to check scores.")


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestMMMUProEval("test_mmmu_pro_evaluation"))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)