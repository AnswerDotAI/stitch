import anthropic
from retry import retry

anthropic_client = anthropic.Anthropic()

@retry(tries=3, delay=5)
def claude_predictor(model_name: str):
    def predictor(question: str):
        return (
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=1000,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": question}]},
                ],
            )
            .content[0]
            .text
        )

    return predictor
