"""Provider factory. This is the ONLY module allowed to import a provider package.

Chat model: selected by LLM_PROVIDER (currently "groq").
Embeddings: always a local sentence-transformers model (no API key), never Groq.
Provider imports are done inside the functions so importing this module is cheap
and does not require any API key.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from attack_mapper import config

SUPPORTED_PROVIDERS = ("groq",)


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return the chat model for the configured provider."""
    provider = config.LLM_PROVIDER
    if not config.LLM_MODEL:
        raise ConfigError("LLM_MODEL is not set. Add it to your .env (see .env.example).")

    if provider == "groq":
        import os

        if not os.getenv("GROQ_API_KEY"):
            raise ConfigError("GROQ_API_KEY is not set. Add it to your .env (see .env.example).")
        from langchain_groq import ChatGroq

        return ChatGroq(model=config.LLM_MODEL, temperature=temperature, max_retries=3)

    raise ConfigError(
        f"Unsupported LLM_PROVIDER={provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def get_embeddings() -> Embeddings:
    """Return a local sentence-transformers embedding model (runs on this machine)."""
    if not config.EMBEDDING_MODEL:
        raise ConfigError("EMBEDDING_MODEL is not set. Add it to your .env (see .env.example).")
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
