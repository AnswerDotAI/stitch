import os
from typing import Literal, List, Callable
from pydantic import BaseModel
import datasets
import srsly
from stitch.llms.utils import get_predictor_fn


class EvalSet(BaseModel):
    dataset: datasets.Dataset
    name: str
    corpus_mapping: dict | None = None
    lang: str = "english"
    noise_format: Literal['ids', 'text', 'no_noise', 'full_document'] = "ids"
    noise_key: str = "noise_ids"
    noise_sample: int | Literal["all"] = "all"
    relevant_format: Literal["ids", "text"] = "ids"
    relevant_key: str = "relevant_doc_ids"
    valid_doc_modes: List[str] = ["relevant_first", "random", "relevant_last", "no_context", "special"]

    class Config:
        arbitrary_types_allowed = True


class Model(BaseModel):
    name: str
    provider: str
    max_context: int
    predictor_fn: Callable = None
    has_special_mode: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.predictor_fn = get_predictor_fn(self.name, self.provider)

    def predict(self, prompt, **kwargs):
        return self.predictor_fn(prompt, **kwargs)


class Template(BaseModel):
    name: str
    doc_separator_end: str = "\n"
    doc_separator_start: str = ""
    role_block: str
    answers_block: str
    document_block: str
    question_block: str
    instruction_block: str

    def _add_lang_role(self, role: str, lang: str):
        extra_instructions = f"The question, potential answers and documents will be given to you in {lang.title()}. As a bilingual expert, you will use your {lang.title()} knowledge to understand them and respond in English."
        if "</role>" in role:
            role = role.replace(" </role>", extra_instructions + " </role>")
        else:
            role += " " + extra_instructions
        return role


    def build_prompt_with_documents(
        self,
        question: str,
        answers: str,
        documents: str,
        question_mode: Literal["before", "after", "repeated"],
        lang: str = "english",
    ) -> str:
        question = question.replace("{", "{{").replace("}", "}}")
        answers = answers.replace("{", "{{").replace("}", "}}")
        documents = documents.replace("{", "{{").replace("}", "}}")

        role_block = self.role_block
        if lang.lower() != "english":
            role_block = self._add_lang_role(role_block, lang)

        question_block = self.question_block.format(question=question)
        answers_block = self.answers_block.format(answers=answers)
        document_block = self.document_block.format(documents=documents)
        instruction_block = self.instruction_block

        if question_mode == "before":
            return "\n\n".join(
                [
                    role_block,
                    question_block,
                    document_block,
                    answers_block,
                    instruction_block,
                ]
            )
        elif question_mode == "after":
            return "\n\n".join(
                [
                    role_block,
                    document_block,
                    question_block,
                    answers_block,
                    instruction_block,
                ]
            )
        elif question_mode == "repeated":
            return "\n\n".join(
                [
                    role_block,
                    question_block,
                    document_block,
                    question_block,
                    answers_block,
                    instruction_block,
                ]
            )

    def build_prompt_without_documents(
        self,
        question: str,
        answers: str,
        lang: str = "english",
    ) -> str:
        role_block = self.role_block
        if lang.lower() != "english":
            role_block = self._add_lang_role(self.role_block, lang)
        question = question.replace("{", "{{").replace("}", "}}")
        answers = answers.replace("{", "{{").replace("}", "}}")
        return f"""{role_block}

{self.question_block.format(question=question)}

{self.answers_block.format(answers=answers)}

{self.instruction_block}"""


class Results(BaseModel):
    dataset: str
    predictor_name: str
    template: str
    question_mode: str
    doc_formatting: str
    accuracy: float
    corrected_accuracy: float
    misformatted: int
    raw_outputs: List[str]

    def export(self, base_path: str = "results/"):
        if not os.path.exists(f"{base_path}{self.predictor_name.replace('/', '-')}"):
            os.makedirs(
                f"{base_path}/{self.predictor_name.replace('/', '-')}", exist_ok=True
            )

        srsly.write_json(
            f"{base_path}/{self.predictor_name.replace('/', '-')}/{self.dataset}_{self.template}_{self.question_mode}_{self.doc_formatting}.json",
            self.dict(),
        )
