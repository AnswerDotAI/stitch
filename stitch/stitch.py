import random
import re
from typing import Optional, Dict, Callable, Literal
from tqdm import tqdm
from stitch.pydantic_objects import Template, Results, EvalSet, Model
from datasets import Dataset


def eval_on_dataset(
    model: Model,
    evalset: EvalSet,
    doc_formatting: Literal[
        "relevant_first", "relevant_last", "random", "only_relevant", "no_context"
    ],
    question_mode: Literal["before", "after", "repeated"],
    template: Template,
    corpus_mapping: Optional[Dict[str, str]] = None,
    export: bool = True,
    export_path: str = "results_questionanswers_before/",
    print_rolling: bool = False,
):
    accurate = 0
    misformatted = 0
    errors = 0
    raw_answers = []
    dataset = evalset.dataset
    lang = evalset.lang

    for q in tqdm(dataset):
        answer_key = "Answer"
        if evalset.relevant_format == "ids" and evalset.noise_format == "ids":
            doc_ids = q[evalset.relevant_key]
            noise_docs = q[evalset.noise_key]
            if doc_formatting not in ["only_relevant", "no_context"]:
                if evalset.noise_sample != "all":
                    random.seed(42)
                    noise_docs = random.sample(noise_docs, evalset.noise_sample)
                if doc_formatting == "relevant_first":
                    doc_ids += noise_docs
                else:
                    doc_ids = noise_docs + doc_ids
                if doc_formatting == "random":
                    random.seed(42)
                    random.shuffle(doc_ids)
            docs = [
                evalset.corpus_mapping[f"{id}_{dataset[0]['subset'].split('_')[0]}"]
                for id in doc_ids
            ]
        elif evalset.relevant_format == "text" and evalset.noise_format == "ids":
            docs = [q[evalset.relevant_key]]
            noise_docs = q[evalset.noise_key]
            if doc_formatting not in ["only_relevant", "no_context"]:
                if doc_formatting == "relevant_first":
                    docs = docs + [evalset.corpus_mapping[f"{id}_{dataset[0]['subset'].split('_')[0]}"] for id in noise_docs]
                else:
                    docs = [evalset.corpus_mapping[f"{id}_{dataset[0]['subset'].split('_')[0]}"] for id in noise_docs] + docs
                    if doc_formatting == "random":
                        random.seed(42)
                        random.shuffle(docs)
        elif evalset.relevant_format == "text" and evalset.noise_format == "full_document":
            docs = (
                [q[evalset.noise_key]]
                if doc_formatting != "only_relevant"
                else [q[evalset.relevant_key]]
            )
        else:
            raise NotImplementedError(f"Combination of relevant_format={evalset.relevant_format} and noise_format={evalset.noise_format} not implemented!")

        if len(docs) > 1:
            document_string = ""
            for i, doc in enumerate(docs, start=1):
                if i > 1:
                    document_string += f"\n{template.doc_separator_start.format(doc_id=i)}{doc}{template.doc_separator_end}"
                else:
                    document_string += f"{template.doc_separator_start.format(doc_id=1)}{doc}{template.doc_separator_end}"
        else:
            document_string = f"{template.doc_separator_start.format(doc_id=1)}\n{docs[0]}\n{template.doc_separator_end}"


        if question_mode == "special" and doc_formatting != "no_context":
            prompt = template.build_prompt_without_documents(
                question=q["question"],
                answers=q["answers_string"],
                lang=lang,
            )
            documents = [{"title": f"Document {i}", "text": doc} for i, doc in enumerate(docs, start=1)]
        elif doc_formatting == "no_context":
            prompt = template.build_prompt_without_documents(
                question=q["question"],
                answers=q["answers_string"],
                lang=lang,
            )
        else:
            prompt = template.build_prompt_with_documents(
                question=q["question"],
                answers=q["answers_string"],
                question_mode=question_mode,
                documents=document_string,
                lang=lang,
            )
        
        # Ensure there are no extra newlines adding noise
        prompt = prompt.replace("\n\n\n", "\n\n")
        try:
            if question_mode == "special":
                print('Special cohere call...', flush=True)
                output = model.predict(prompt.strip(), documents=documents)
            else:
                output = model.predict(prompt.strip())
        except Exception as e:
            if errors > 8:
                print("CHAIN ERROR -- PLEASE CHECK YOUR LOGS TO VERIFY IF A DEEPER ISSUE IS OCCURING")
                print("CHAIN ERROR -- PLEASE CHECK YOUR LOGS TO VERIFY IF A DEEPER ISSUE IS OCCURING")
            print(e, flush=True)
            print("Provider Error, likely context length.")
            errors += 1
            raw_answers.append(str(e))
            continue

        output = output.replace('*', '').replace('_', '')

        raw_answers.append(output)
        search_result = re.search(answer_key + r"\s*:\s*([A-I])", output)
        if search_result is None:
            misformatted += 1
            print(prompt)
            print("MISFORMED OUTPUT, skipping...")
            print(output, flush=True)
            continue

        if q["correct_letter"] == search_result.group(1):
            accurate += 1

        if print_rolling:
            print("Correct letter:", q["correct_letter"], flush=True)
            print("Predicted letter:", search_result.group(1), flush=True)
            print(f"Rolling Accuracy: {accurate / len(raw_answers)}", flush=True)

    accuracy = accurate / len(dataset)
    corrected_accuracy = accurate / max((len(dataset) - errors - misformatted), 1)

    results = Results(
        dataset=evalset.name,
        predictor_name=model.name,
        template=template.name,
        question_mode=question_mode,
        doc_formatting=doc_formatting,
        accuracy=accuracy,
        corrected_accuracy=corrected_accuracy,
        misformatted=misformatted,
        raw_outputs=raw_answers,
    )

    if export:
        results.export(base_path=export_path)

    return results
