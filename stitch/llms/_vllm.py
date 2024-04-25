from retry import retry
from openai import OpenAI

vllm_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

@retry(tries=3, delay=5)
def vllm_predictor(model_name: str):
    def predictor(question: str):
        return (
            vllm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": question}],
                max_tokens=1000,
                temperature=0.0,
            )
            .choices[0]
            .message
            .content
        )

    return predictor
