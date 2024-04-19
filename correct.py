# TODO: CLEANUP

# %%
import srsly
import glob

# %%
models = []

for folder in glob.glob('./results/*'):
    model_name = folder.split('/')[-1]
    models.append(model_name)

# %%
from datasets import load_dataset
import dotenv

dotenv.load_dotenv()

from typing import Dict
from datasets import load_dataset, Dataset

def load_evalset(dataset_name: str) -> Dataset:
    return load_dataset("bclavie/STITCH", dataset_name + "_questions", split="test")


def get_corpus_mapping(dataset_name: str) -> Dict[str, str]:
    dataset = load_dataset("bclavie/STITCH", ds_name + "_corpus", split="test")
    return {doc["id"]: doc["content"] for doc in dataset}


dataset_names = {"biomrc_control": "biomrc_known",
"biomrc_hard": "biomrc",
"bsard_hard": "bsard",
"bsard_control": "bsard_known",
"proxima_hard": "proxima"}

corpus_names = {"biomrc_control": "biomrc",
"biomrc_hard": "biomrc",
"bsard_hard": "bsard",
"bsard_control": "bsard_known",
"proxima_hard": "proxima"}

datasets, corpuses = {}, {}

for ds_name, ds in dataset_names.items():
    datasets[ds_name] = load_evalset(ds)

from tqdm import tqdm
import re
import os

os.makedirs('fixed_results/test_model', exist_ok=True)
from collections import defaultdict

# results_for_model = []
# attributes = defaultdict(set)
# hard_datasets = ['proxima_hard', 'biomrc_hard', 'bsard_hard']
# control_datasets = ['biomrc_control', 'bsard_control']
# datasets = hard_datasets + control_datasets
# doc_formatting = ["relevant_first", "relevant_last", "random", "only_relevant", "no_context"]
# special_formatting = ["special"]
# question_mode = ["before", "after", "repeated"]

# results = []

misformatted = 0
total = 0

import anthropic
from retry import retry
anthropic_client = anthropic.Anthropic()

@retry(tries=3, delay=5)
def claude_correct(prompt):
    return (
        anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            system="You are a helpful AI assistant. Your specialty is data extraction. You do not modify the user's original reasoning or intent at all, and always extract to the best of your ability.",
            max_tokens=1000,
            temperature=0.0,
            messages=prompt,
        )
        .content[0]
        .text
    )

print('loaded!')

for file in tqdm(glob.glob('./results/**/*.json')):
    print(file)
    if 'gemma' in file or 'reka' in file:
        continue
    fixed_file_path = "/".join(file.split('/')[:-1]).replace("results", "fixed_results")
    filename = file.split('/')[-1]
    os.makedirs(fixed_file_path, exist_ok=True)
    loaded = srsly.read_json(file)
    fixed = 0
    dataset = datasets[loaded['dataset']]
    full_answers = []
    for i, entry in enumerate(loaded['raw_outputs']):
        search_result = re.search("Answer" + r"\s*:\s*([A-I])", entry)
        if search_result is not None:
            full_answers.append(search_result.group(1))
            continue
        answers = dataset[i]['answers_string']

        # prompt = [{"role": "user", "content":RAW_PROMPT.format(question=question, answers=answers)}]
        prompt = [{"role": "user", "content": f"""We had a contest where users had to write their answer in <answer></answer> tags, but some of them got distracted and ended up just writing free text, or their reasoning, but not their final answer.\nYour role is to read their answer and parse their answer to the right format.\n\nThe possible answers were:\n\n<potential_answers>\n{answers}\n</potential_answers>\n\nThe user's entry was:\n<entry>\n{entry}\n</entry>

Write the answer key (A, B, C, D, E, F, G or I) in the <answer></answer> tags and nothing else. If no answer can be extracted from the text, you will write <answer>None</answer>."""}]
        prompt += [{"role": "assistant", "content": "<answer>"}]
        answer = claude_correct(prompt)
        full_answers.append(answer.split('<')[0])
    loaded['answers'] = full_answers
    assert len(full_answers) == 54
    srsly.write_json(fixed_file_path + '/' + filename, loaded)
# %%



print('done')