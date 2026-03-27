import torch
from transformers import AutoModelForSequenceClassification
from lossyformer.main import LossyFormer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "prajjwal1/bert-tiny"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Take an initialised, but untrained bert-tiny model, and pass that into the LossyFormer.

dataset_kwargs = {
    "dataset_name": "imdb",
    "dataset_config": None,
    "tokenizer_name": model_name,
    "text_columns": ["text"],
    "max_length": 128,
    "train_batch_size": 32,
    "eval_batch_size": 64,
}

# Define the kwargs as above, to give the dataset the user wants to execute lossy.fit on, as well as the batch sizes they want to use for both dataloaders. The dataloaders are constructed inside the LossyFormer.

lossy = LossyFormer(
    model_name=model_name,
    allowed_accuracy_loss=0.015,
    device=device,
    dataset_kwargs=dataset_kwargs,
)

optimized_model = lossy.fit(model, max_ft_steps=500)

if optimized_model is not None:
    print(f"Success! Model optimized with Early Exit Threshold: {optimized_model.threshold}")
    
    pure_encoder = getattr(optimized_model, "base_model", optimized_model)
    
    # In a real scenario, you could save it here
    # pure_encoder.save_pretrained("./my-lossy-model")
else:
    print("Pruning failed to meet the strict accuracy threshold.")