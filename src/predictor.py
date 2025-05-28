# src/predictor.py

import ollama
import re
import json
from typing import Dict, Any, List


def predict_home_run_probabilities(context: List[Dict[str, Any]], model: str = "llama3") -> Dict[str, Any]:
    """
    Sends a prompt to the Ollama LLM asking for probabilities that batters will hit home runs.

    Parameters:
        context (List[Dict[str, Any]]): Array of JSON objects containing player, pitcher, and game data.
        model (str): Name of the locally installed Ollama model to use (e.g., 'llama3').

    Returns:
        Dict with structured prediction data for all players.
    """
    # Convert JSON array to formatted string
    context_str = json.dumps(context, indent=2)

    prompt = (
        f"{context_str}\n\n"
        "You are an expert baseball statistician. "
        "The JSON data above contains information about baseball players scheduled to play today, "
        "including batting stats, pitcher matchups, recent trends, and ballpark factors. "
        "Ballpark factors above 100 favor home runs; below 100 suppress them. "
        "For each player, calculate the probability (0-100%) that they will hit a home run and prepare a brief "
        "reasoning."
        "Format your response as:\n"
        "Player Name: X% - reason\n"
        "Order by highest to lowest probability."
    )

    try:
        print("Asking Ollama...")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = response['message']['content'].strip()

        # Extract all player predictions from response
        predictions = []
        lines = content.split('\n')

        for line in lines:
            # Match patterns like "Player Name: 75% - reasoning"
            match = re.search(r'([^:]+):\s*(\d+(?:\.\d+)?)\s*%\s*-\s*(.+)', line.strip())
            if match:
                predictions.append({
                    "player": match.group(1).strip(),
                    "probability": float(match.group(2)),
                    "reasoning": match.group(3).strip()
                })

        return {
            "predictions": predictions,
            "raw_response": content,
            "total_players": len(context),
            "parsed_players": len(predictions)
        }

    except Exception as e:
        return {
            "predictions": [],
            "error": str(e),
            "raw_response": None,
            "total_players": len(context) if context else 0,
            "parsed_players": 0
        }