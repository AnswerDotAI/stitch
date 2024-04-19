import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def transformers_predictor(model_name: str):
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def predictor(question: str):
        messages = [
            {"role": "user", "content": question},
        ]
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        generation_args = {
            "max_new_tokens": 1000,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = pipe(messages, **generation_args)
        return output[0]['generated_text']

    return predictor


