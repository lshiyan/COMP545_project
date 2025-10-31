from openai import OpenAI

_client = None

def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client