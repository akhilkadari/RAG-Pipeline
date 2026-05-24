"""Tiny abstraction over chat-completions for OpenAI / Anthropic.

We mostly need:
  * chat(prompt, system) -> str
  * json(prompt, system) -> dict   (assumes the model returns JSON)
"""
from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

from src.config import settings

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.provider = (provider or settings.llm_provider).lower()
        self.model = model or settings.generation_model

        if self.provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._openai = OpenAI(api_key=settings.openai_api_key)
            self._anthropic = None
        elif self.provider == "anthropic":
            if not settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
            self._openai = None
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.provider == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._openai.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        # Anthropic
        resp = self._anthropic.messages.create(  # type: ignore[union-attr]
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        if not resp.content:
            return ""
        return resp.content[0].text.strip()

    def json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        text = self.chat(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        return _coerce_json(text)
