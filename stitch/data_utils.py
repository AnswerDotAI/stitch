from typing import Dict
from datasets import load_dataset, Dataset


def load_evalset(repo_name: str, dataset_name: str, split: str) -> Dataset:
    return load_dataset(repo_name, dataset_name, split=split)


def get_corpus_mapping(repo_name: str, dataset_name: str, split: str) -> Dict[str, str]:
    dataset = load_dataset(repo_name, dataset_name, split=split)
    return {doc["id"]: doc["content"] for doc in dataset}
