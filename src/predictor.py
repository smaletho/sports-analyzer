# src/predictor.py

import ollama
import re
from typing import Dict, Any

def predict_home_run_probability(context: str, model: str = "llama3") -> Dict[str, Any]:
    """
    Sends a prompt to the Ollama LLM asking for a probability that the batter will hit a home run.

    Parameters:
        context (str): A natural-language summary of the player, pitcher, and game conditions.
        model (str): Name of the locally installed Ollama model to use (e.g., 'llama3').

    Returns:
        Dict with structured prediction data.
    """
    prompt = (
        f"{context}\n\n"
        "Based on the data above, what is the likelihood (as a percentage between 0 and 100) that this player will "
        "hit a home run in this game? Provide only a number followed by a brief reason."
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = response['message']['content'].strip()

        # Extract probability from response
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', content)
        probability = float(match.group(1)) if match else None

        return {
            "probability": probability,
            "raw_response": content
        }

    except Exception as e:
        return {
            "probability": None,
            "error": str(e),
            "raw_response": None
        }
