"""
src/inference/ollama_client.py

Ollama local inference wrapper for the fine-tuned AeroSense ChartQA model.
"""

from __future__ import annotations

import requests
from loguru import logger

ALPACA_PROMPT = """Below is an instruction related to aeronautical charts and aviation procedures.
Write a response that accurately answers the question.

### Instruction:
{instruction}

### Input:
{context}

### Response:
"""


class OllamaClient:
    """Wraps Ollama REST API for local inference."""

    def __init__(
        self,
        model: str = "aerosense-chartqa",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._check_connection()

    def generate(
        self,
        question: str,
        context: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,   # Low temp for factual aviation answers
    ) -> str:
        """Generate a response for a given aviation question."""
        prompt = ALPACA_PROMPT.format(instruction=question, context=context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["### Instruction:", "### Input:"],
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def _check_connection(self) -> None:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if self.model not in models:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}. "
                    f"Run: ollama create {self.model} -f models/Modelfile"
                )
            else:
                logger.info(f"Ollama connected ✓  |  Model '{self.model}' ready")
        except Exception as e:
            logger.warning(f"Ollama not reachable at {self.base_url}: {e}")
