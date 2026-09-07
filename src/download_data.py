from datasets import load_dataset

dataset = load_dataset(
    "qiaojin/PubMedQA",
    "pqa_labeled", 
    split="train"
)

dataset = dataset.select(range(500))  # Select the first 500 samples for demonstration purposes

dataset.to_json("data/pubmedqa.jsonl")

print(f"Saved {len(dataset)} samples to data/pubmedqa.jsonl")