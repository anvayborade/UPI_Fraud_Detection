from __future__ import annotations

import os

from openai import OpenAI

from src.intelligence.complaint_schema import ScamIntelligence


class ComplaintLLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def extract(self, redacted_text: str) -> ScamIntelligence:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Extract structured digital-payment scam intelligence. Do not invent identifiers. "
                        "Use UNKNOWN when evidence is insufficient. The text has already been redacted."
                    ),
                },
                {"role": "user", "content": redacted_text},
            ],
            text_format=ScamIntelligence,
        )
        if response.output_parsed is None:
            raise ValueError("The model did not return a parsed structured output")
        return response.output_parsed
