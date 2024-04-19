import os
from retry import retry
from together import Together

together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

@retry(tries=3, delay=5)
def together_predictor(model_name: str):
    def predictor(question: str):
        return (
            together_client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[{"role": "user", "content": question}],
            )
            .choices[0]
            .message.content
        )

    return predictor
