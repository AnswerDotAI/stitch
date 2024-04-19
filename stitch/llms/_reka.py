import os
import reka
from retry import retry

reka.API_KEY = os.environ.get("REKA_API_KEY")

@retry(tries=3, delay=5)
def reka_predictor(model_name: str):
    def predictor(question: str):
        return reka.chat(question, temperature=0., model_name=model_name)['text']

    return predictor