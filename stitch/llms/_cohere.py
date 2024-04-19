import cohere
from retry import retry


co = cohere.Client()

@retry(tries=3, delay=5)
def cohere_predictor(model_name: str):
    def predictor(question: str, documents: None | list[dict] = None):
        if documents:
            return co.chat(message=question, model=model_name, temperature=0., documents=documents).text
        return co.chat(message=question, model=model_name, temperature=0.).text


    return predictor

