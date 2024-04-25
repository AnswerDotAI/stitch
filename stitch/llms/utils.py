from typing import Callable
from stitch.llms._anthropic import claude_predictor
from stitch.llms._openai import openai_predictor
from stitch.llms._together import together_predictor
from stitch.llms._reka import reka_predictor
from stitch.llms._cohere import cohere_predictor
from stitch.llms._hf_transformers import transformers_predictor
from stitch.llms._vllm import vllm_predictor

def get_predictor_fn(model_name: str, provider: str) -> Callable:
    match provider:
        case "together":
            return together_predictor(model_name)
        case "reka":
            return reka_predictor(model_name)
        case "openai":
            return openai_predictor(model_name)
        case "claude":
            return claude_predictor(model_name)
        case "cohere":
            return cohere_predictor(model_name)
        case "hf_transformers":
            return transformers_predictor(model_name)
        case "vllm":
            return vllm_predictor(model_name)
        case other:
            raise ValueError(f"Unknown provider: {other}")
