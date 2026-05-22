"""
src/prompts.py

Shared Alpaca prompt template for training and inference.
Single source of truth — prevents format drift between the two.
"""

from __future__ import annotations

ALPACA_PROMPT = """Below is an instruction related to aeronautical charts and aviation procedures.
Write a response that accurately answers the question.

### Instruction:
{instruction}

### Input:
{context}

### Response:
{response}"""


def format_for_training(instruction: str, context: str, response: str) -> str:
    """Fill all three slots — used by SFTTrainer so the model learns the full pattern."""
    return ALPACA_PROMPT.format(instruction=instruction, context=context, response=response)


def format_for_inference(instruction: str, context: str = "") -> str:
    """Fill only instruction + context, leave Response blank for the model to complete."""
    return ALPACA_PROMPT.format(instruction=instruction, context=context, response="")
