import os
from typing import Optional

from groq import Groq


DEFAULT_MODEL = "llama-3.1-8b-instant"


def load_overview_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def run_groq_agent(
    user_query: str,
    overview_text: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    extra_context: Optional[str] = None,
) -> str:
    if not user_query:
        return ""

    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    system_msg = (
        "You are a concise assistant embedded in a rare-disease protein analysis dashboard. "
        "Use the provided knowledge base context when relevant. If the answer is not in the "
        "context, say so and provide best-effort guidance."
    )
    if overview_text:
        system_msg += f"\n\nKnowledge Base:\n{overview_text}"
    if extra_context:
        system_msg += f"\n\nCurrent Analysis Context:\n{extra_context}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_query},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content if resp and resp.choices else ""
