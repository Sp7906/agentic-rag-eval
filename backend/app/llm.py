"""Thin Azure OpenAI wrapper. Everything works without it (offline mode)."""
from openai import AzureOpenAI

from . import config

_client = None


def enabled() -> bool:
    return bool(config.AOAI_ENDPOINT and config.AOAI_KEY)


def _c() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(azure_endpoint=config.AOAI_ENDPOINT, api_key=config.AOAI_KEY,
                              api_version=config.AOAI_VERSION)
    return _client


def chat(messages: list[dict]) -> str:
    r = _c().chat.completions.create(model=config.AOAI_DEPLOYMENT, messages=messages, temperature=0)
    return (r.choices[0].message.content or "").strip()


def embed(texts: list[str]) -> list[list[float]]:
    r = _c().embeddings.create(model=config.AOAI_EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in r.data]
