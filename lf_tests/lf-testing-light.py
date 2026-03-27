import argparse
import logging
import torch
import csv
import gc
from transformers import AutoModelForSequenceClassification
from lossyformer.main import LossyFormer
from lossyformer.utils import eval_accuracy, eval_speed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser(description="Run LossyFormer Single Pass")
parser.add_argument(
    "--model", type=str, choices=["bert-base", "roberta", "bert-tiny"], required=True
)
parser.add_argument("--target_drop", type=float, default=0.03, help="Allowed accuracy drop (e.g., 0.03 for 3%)")
parser.add_argument("--max_ft_steps", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=1)
parser.add_argument("--step_keep_ratio", type=float, default=0.90)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BASE_PATH = "models"

IMDB_DATASET_KWARGS = {
    "dataset_name": "imdb",
    "dataset_config": None,
    "tokenizer_name": "prajjwal1/bert-tiny",
    "text_columns": ["text"],
    "max_length": 128,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "num_labels": 2,
}

def get_mnli_dataset_kwargs(checkpoint_name):
    return {
        "dataset_name": "glue",
        "dataset_config": "mnli",
        "tokenizer_name": checkpoint_name,
        "text_columns": ["premise", "hypothesis"],
        "max_length": 128,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "num_labels": 3,
    }

MODEL_CONFIG = {
    "bert-base": {
        "checkpoint_name": "bert-base-uncased",
        "baseline_checkpoint": f"{BASE_PATH}/bert-base-glue-mnli-baseline",
        "dataset_kwargs": get_mnli_dataset_kwargs("bert-base-uncased"),
    },
    "roberta": {
        "checkpoint_name": "roberta-base",
        "baseline_checkpoint": f"{BASE_PATH}/roberta-base-glue-mnli-baseline",
        "dataset_kwargs": get_mnli_dataset_kwargs("roberta-base"),
    },
    "bert-tiny": {
        "checkpoint_name": "prajjwal1/bert-tiny",
        "baseline_checkpoint": f"{BASE_PATH}/bert-tiny-imdb-baseline",
        "dataset_kwargs": IMDB_DATASET_KWARGS,
    },
}

checkpoint_name = MODEL_CONFIG[args.model]["checkpoint_name"]
baseline_checkpoint = MODEL_CONFIG[args.model]["baseline_checkpoint"]
output_csv = f"{args.model}_single_run_results.csv"
dataset_kwargs = MODEL_CONFIG[args.model]["dataset_kwargs"]

print(f"\n" + "=" * 50)
print(f"LOSSYFORMER RUN: {args.model.upper()} | TARGET DROP: {args.target_drop * 100:.1f}%")
print("=" * 50)

print(f"Loading weights from {baseline_checkpoint}.pt...")
state_dict = torch.load(f"{baseline_checkpoint}.pt", map_location=DEVICE, weights_only=False)
if hasattr(state_dict, "state_dict"):
    state_dict = state_dict.state_dict()

current_model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint_name, num_labels=dataset_kwargs["num_labels"]
)
current_model.load_state_dict(state_dict, strict=False)

lossy = LossyFormer(
    model_name=args.model,
    allowed_accuracy_loss=args.target_drop,
    device=DEVICE,
    max_iterations=args.max_iterations,
    step_keep_ratio=args.step_keep_ratio,
    dataset_kwargs=dataset_kwargs,
)

opt_model = lossy.fit(current_model, max_ft_steps=args.max_ft_steps)

if opt_model is not None:
    pure_pruned_model = opt_model.original_model if hasattr(opt_model, "original_model") else opt_model

    print(f"\n--- Final Evaluation ---")
    acc = eval_accuracy(opt_model, lossy._eval_loader, DEVICE)
    _, lat = eval_speed(pure_pruned_model, lossy._eval_loader, DEVICE, n=50, warmup=5)
    params = sum(p.numel() for p in pure_pruned_model.parameters()) / 1e6
    threshold = getattr(opt_model, "threshold", "N/A")
    final_reduction = lossy.iteration_history[-1]["percent_pruned"] if lossy.iteration_history else 0.0

    print(f"  Acc: {acc * 100:.2f}% | Latency: {lat * 1000:.2f}ms | Params: {params:.2f}M | Threshold: {threshold}")

    fieldnames = ["Model", "Target Drop (%)", "Accuracy (%)", "Latency (ms)", "Params (M)", "EE Threshold", "Total Reduction (%)"]
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "Model": args.model,
            "Target Drop (%)": args.target_drop * 100,
            "Accuracy (%)": round(acc * 100, 2),
            "Latency (ms)": round(lat * 1000, 2),
            "Params (M)": round(params, 2),
            "EE Threshold": round(threshold, 2) if isinstance(threshold, float) else threshold,
            "Total Reduction (%)": round(final_reduction, 2),
        })
    print(f"\nRun complete! Result saved to {output_csv}")
else:
    print("Optimization failed or returned None.")

del lossy, opt_model, current_model
gc.collect()
torch.cuda.empty_cache()