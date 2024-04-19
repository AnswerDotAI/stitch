from retry import retry
import openai

openai_client = openai.AzureOpenAI()

@retry(tries=3, delay=5)
def openai_predictor(model_name: str):
    def predictor(question: str):
        return (
            openai_client.chat.completions.create(
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
