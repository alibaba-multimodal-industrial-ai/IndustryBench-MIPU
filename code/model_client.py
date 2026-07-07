"""LLM API abstraction for OpenAI-compatible and Anthropic endpoints."""

import asyncio
import base64
import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


DEFAULT_OPENAI_MAX_TOKENS = 8192
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192


def _encode_image(path: Path) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


class ModelClient(ABC):
    """Base class for LLM API clients."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        images: list[Path] | None = None,
    ) -> str:
        """Send a chat request and return the text response."""
        ...

    async def chat_with_finish(
        self,
        messages: list[dict[str, Any]],
        images: list[Path] | None = None,
    ) -> tuple[str, str]:
        """Like chat() but also returns the finish_reason."""
        text = await self.chat(messages, images=images)
        return text, "unknown"


class OpenAIClient(ModelClient):
    """Client for OpenAI-compatible APIs (also covers Qwen/DashScope, vLLM, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_thinking: bool = False,
        max_tokens: int = DEFAULT_OPENAI_MAX_TOKENS,
    ):
        self.model = model
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("API_KEY", ""),
            base_url=base_url or os.environ.get("API_BASE_URL"),
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        images: list[Path] | None = None,
    ) -> str:
        text, _ = await self.chat_with_finish(messages, images=images)
        return text

    async def chat_with_finish(
        self,
        messages: list[dict[str, Any]],
        images: list[Path] | None = None,
    ) -> tuple[str, str]:
        api_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user" and images:
                parts: list[dict] = []
                for img_path in images:
                    b64, mime = await asyncio.to_thread(_encode_image, img_path)
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                parts.append({"type": "text", "text": content})
                api_messages.append({"role": role, "content": parts})
                images = None
            else:
                api_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
        }
        if self.enable_thinking:
            kwargs["extra_body"] = {"enable_thinking": True}

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return (choice.message.content or ""), (choice.finish_reason or "unknown")


class AnthropicClient(ModelClient):
    """Client for the Anthropic Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        thinking: bool = False,
        thinking_budget: int = 10000,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.client = AsyncAnthropic(
            api_key=api_key or os.environ.get("API_KEY", ""),
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        images: list[Path] | None = None,
    ) -> str:
        system_prompt = None
        api_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_prompt = content
                continue

            if role == "user" and images:
                parts: list[dict] = []
                for img_path in images:
                    b64, mime = await asyncio.to_thread(_encode_image, img_path)
                    parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    })
                parts.append({"type": "text", "text": content})
                api_messages.append({"role": role, "content": parts})
                images = None
            else:
                api_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self.thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        response = await self.client.messages.create(**kwargs)
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""


def create_client(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    enable_thinking: bool = False,
    max_tokens: int | None = None,
) -> ModelClient:
    """Factory function to create a ModelClient from CLI arguments."""
    if provider == "anthropic":
        return AnthropicClient(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS,
            thinking=enable_thinking,
        )
    return OpenAIClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens or DEFAULT_OPENAI_MAX_TOKENS,
    )
