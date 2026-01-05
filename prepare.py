import os
import json
from datasets import load_dataset

os.makedirs("datasets", exist_ok=True)
os.makedirs("datasets/LongEmbed", exist_ok=True)

datasets = ["narrativeqa", "summ_screen_fd", "qmsum", "2wikimqa", "passkey", "needle"]
for dataset in datasets:
    corpus = load_dataset(path="dwzhu/LongEmbed", name=dataset, split="corpus")
    queries = load_dataset(path="dwzhu/LongEmbed", name=dataset, split="queries")
    qrels = load_dataset(path="dwzhu/LongEmbed", name=dataset, split="qrels")

    corpus = [x for x in corpus]
    queries = [x for x in queries]
    qrels = [x for x in qrels]

    with open(f"datasets/LongEmbed/{dataset}/corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)

    with open(f"datasets/LongEmbed/{dataset}/queries.json", "w") as f:
        json.dump(queries, f, indent=2)

    with open(f"datasets/LongEmbed/{dataset}/qrels.json", "w") as f:
        json.dump(qrels, f, indent=2)