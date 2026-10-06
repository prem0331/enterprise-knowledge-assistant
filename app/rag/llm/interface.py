"""LLM provider abstraction.

This interface allows swapping LLM implementations without
changing the rest of the application.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from groq.types.chat import ChatCompletionMessageParam
else:
    ChatCompletionMessageParam = dict[str, str]


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt/message.
            system_prompt: Optional system prompt.
            **kwargs: Additional provider-specific parameters (temperature, max_tokens, etc.).

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def generate_with_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> str:
        """Generate a response from a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated text response.
        """
        ...
