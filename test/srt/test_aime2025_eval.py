import os
import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_SCORE_THRESHOLDS = {
    # 可自定义每个模型的分数阈值
    # "meta-llama/Llama-3-8B-Instruct": 0,
    # "Qwen/Qwen1.5-7B-Chat": 0,
    "Qwen/QwQ-32B": 0,
    # "XiaomiMiMo/MiMo-7B-SFT": 0,
}

DEFAULT_MODELS = [
    # "meta-llama/Llama-3-8B-Instruct",
    # "Qwen/Qwen1.5-7B-Chat",
    "Qwen/QwQ-32B",
    # "XiaomiMiMo/MiMo-7B-SFT",
]

DEFAULT_SCORE_TYPE = "pass@1"  # 可选 pass@1, pass@k, mean
DEFAULT_K = 5
DEFAULT_NUM_SAMPLES = 5
DEFAULT_MAX_EXAMPLES = 20  # 调试用，实际可设大
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_models(model_string):
    return [model.strip() for model in model_string.split(",") if model.strip()]


def load_aime2025():
    ds = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    return ds


def get_answer(example):
    return str(example["answer"]).strip()


def score_pass_at_k(preds, gold, k):
    for pred in preds[:k]:
        if pred.strip() == gold:
            return 1
    return 0

def score_mean(preds, gold):
    return sum([1 if pred.strip() == gold else 0 for pred in preds]) / len(preds)


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
    if failed_models:
        raise AssertionError("\n".join(failed_models))


def evaluate_model(model_name, ds, score_type, k, num_samples, max_examples, device):
    print(f"\n加载模型 {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device.startswith("cuda") else -1)
    results = []
    correct = 0
    total = 0
    for i, example in enumerate(ds):
        if i >= max_examples:
            break
        question = "Please reason step by step, and put your final answer within \boxed{}. " + example["question"]
        gold = get_answer(example)
        prompt = question.strip() + "\nAnswer:"
        preds = []
        if score_type == "pass@1":
            output = pipe(prompt, max_new_tokens=16, do_sample=False)[0]["generated_text"]
            print("outputttttttttt: ", output)
            pred = output.split("Answer:")[-1].strip().split("\n")[0]
            preds = [pred]
            score = 1 if pred == gold else 0
        elif score_type == "pass@k":
            for _ in range(k):
                output = pipe(prompt, max_new_tokens=16, do_sample=True, temperature=0.7)[0]["generated_text"]
                pred = output.split("Answer:")[-1].strip().split("\n")[0]
                preds.append(pred)
            score = score_pass_at_k(preds, gold, k)
        elif score_type == "mean":
            for _ in range(num_samples):
                output = pipe(prompt, max_new_tokens=16, do_sample=True, temperature=0.7)[0]["generated_text"]
                pred = output.split("Answer:")[-1].strip().split("\n")[0]
                preds.append(pred)
            score = score_mean(preds, gold)
        else:
            raise ValueError("未知评分方式")
        results.append({
            "question": question,
            "gold": gold,
            "preds": preds,
            "score": score
        })
        correct += score
        total += 1
        if (i+1) % 5 == 0:
            print(f"已评测 {i+1} 题，当前准确率：{correct/total:.3f}")
    final_score = correct / total if total > 0 else 0.0
    print(f"模型 {model_name} 最终得分：{final_score:.4f}")
    return {
        "score": final_score,
        "details": results
    }


class TestAIME2025Eval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (parse_models(",".join(DEFAULT_MODELS)), DEFAULT_SCORE_TYPE, DEFAULT_K, DEFAULT_NUM_SAMPLES, DEFAULT_MAX_EXAMPLES, DEFAULT_DEVICE),
        ]
        cls.ds = load_aime2025()

    def test_aime2025_all_models(self):
        all_results = []
        is_first = True
        for model_group, score_type, k, num_samples, max_examples, device in self.model_groups:
            for model in model_group:
                with self.subTest(model=model):
                    metrics = evaluate_model(model, self.ds, score_type, k, num_samples, max_examples, device)
                    write_results_to_json(model, metrics, "w" if is_first else "a")
                    is_first = False
                    all_results.append((model, metrics["score"]))
        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"Error reading results.json: {e}")
        check_model_scores(all_results)


if __name__ == "__main__":
    unittest.main() 