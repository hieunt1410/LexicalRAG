import os
import json
from datasets import load_dataset

os.makedirs("datasets", exist_ok=True)
os.makedirs("datasets/longembed", exist_ok=True)
os.makedirs("datasets/litsearch", exist_ok=True)
os.makedirs("datasets/mldr", exist_ok=True)

datasets = ["narrativeqa", "summ_screen_fd", "qmsum", "2wikimqa", "passkey", "needle"]
for dataset in datasets:
    corpus = load_dataset(path="dwzhu/LongEmbed", name=dataset, split="corpus")
    queries = load_dataset(path="dwzhu/LongEmbed", name=dataset, split="queries")
    qrels = load_dataset(path="dwzhu/LongEmbed", name=dataset, split="qrels")

    os.makedirs(f"datasets/longembed/{dataset}", exist_ok=True)
    
    corpus = [x for x in corpus]
    queries = [x for x in queries]
    qrels = [x for x in qrels]
    
    for query in queries:
        query["doc_id"] = [x["doc_id"] for x in qrels if x["qid"] == query["qid"]]

    with open(f"datasets/longembed/{dataset}/corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)

    with open(f"datasets/longembed/{dataset}/queries.json", "w") as f:
        json.dump(queries, f, indent=2)

    # with open(f"datasets/LongEmbed/{dataset}/qrels.json", "w") as f:
    #     json.dump(qrels, f, indent=2)
        
# LitSearch
query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
corpus_clean_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")

with open("datasets/litsearch/queries.json", "w") as f:
    query_data = [x for x in query_data]
    json.dump(query_data, f, indent=2)

with open("datasets/litsearch/corpus.json", "w") as f:
    corpus_clean_data = [x for x in corpus_clean_data]
    json.dump(corpus_clean_data, f, indent=2)

# MLDR
# mldr_data = load_dataset("princeton-nlp/MLDR", "mldr", split="full")
# test_data = load_dataset("princeton-nlp/MLDR", "test", split="full")

# with open("datasets/mldr/corpus.json", "w") as f:
#     json.dump(mldr_data, f, indent=2)

# with open("datasets/mldr/test.json", "w") as f:
#     json.dump(test_data, f, indent=2)