"""Groq LLM provider.

Uses Groq API for fast inference with open-source models like Llama 3.1.
Requires GROQ_API_KEY in environment.
"""

import logging
from typing import Any

from groq import Groq
from groq.types.chat import ChatCompletionMessageParam

from app.config.settings import settings
from app.rag.llm.interface import LLMProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Reduce groq logging noise
logging.getLogger("groq").setLevel(logging.WARNING)


class GroqLLMProvider(LLMProvider):
    """LLM provider using Groq API.

    Supports GPT-OSS models and other open-source models available on Groq.
    Model is configurable via GROQ_MODEL environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        **default_kwargs: Any,
    ):
        """Initialize the Groq LLM provider.

        Args:
            api_key: Groq API key. Defaults to settings.groq_api_key.
            model: Model name. Defaults to settings.groq_model.
            **default_kwargs: Default generation parameters (temperature, max_tokens, etc.).
        """
        self._api_key = api_key or settings.groq_api_key
        self._model = model or settings.groq_model
        self._default_kwargs = default_kwargs
        self._client: Groq | None = None

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def _ensure_client(self) -> Groq:
        """Lazy-create the Groq client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. Please set it in .env or pass api_key."
                )
            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt/message.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            Generated text response.
        """
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_with_messages(messages, **kwargs)

    def generate_with_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> str:
        """Generate a response from a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            Generated text response.
        """
        client = self._ensure_client()

        # Merge default kwargs with call-specific kwargs
        params = {**self._default_kwargs, **kwargs}

        logger.debug(
            "Calling Groq API",
            extra={
                "model": self._model,
                "message_count": len(messages),
                "params": {k: v for k, v in params.items() if k != "api_key"},
            },
        )

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                **params,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Groq returned empty response")
            return str(content).strip()
        except Exception as e:
            logger.error("Groq API error", extra={"error": str(e)})
            raise
