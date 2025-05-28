# homeruns/predictor.py

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


def predict_single_player_home_run(player_context: Dict[str, Any], model: str = "llama3") -> Dict[str, Any]:
    """
    Predicts the likelihood of a single player hitting a home run.

    Parameters:
        player_context (Dict[str, Any]): JSON object containing player, pitcher, and game data.
        model (str): Name of the locally installed Ollama model to use (e.g., 'llama3').

    Returns:
        Dict with prediction data for the player.
    """
    # Convert JSON to formatted string
    context_str = json.dumps(player_context, indent=2)

    prompt = (
        f"{context_str}\n\n"
        "You are an expert baseball statistician. "
        "The JSON data above contains information about a baseball player scheduled to play today, "
        "including batting stats, pitcher matchup, recent trends, and ballpark factors. "
        "Ballpark factors above 100 favor home runs; below 100 suppress them. "
        "Calculate the probability (0-100%) that this player will hit a home run today and provide a "
        "detailed explanation of your reasoning. Consider all factors including: "
        "- The player's recent performance and home run trends "
        "- The opposing pitcher's stats and vulnerability to home runs "
        "- The ballpark factor and how it affects home run probability "
        "- Historical matchup data between this batter and pitcher "
        "- Team trends and other contextual factors\n\n"
        "Format your response as:\n"
        "Probability: X%\n"
        "Reasoning: Your detailed explanation here"
    )

    try:
        print(f"Asking Ollama about {player_context.get('player_name', 'unknown player')}...")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = response['message']['content'].strip()

        # Extract probability and reasoning
        prob_match = re.search(r'Probability:\s*(\d+(?:\.\d+)?)\s*%', content, re.IGNORECASE)
        reasoning_match = re.search(r'Reasoning:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        
        probability = float(prob_match.group(1)) if prob_match else 0.0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else content
        
        return {
            "player": player_context.get("player_name", "Unknown"),
            "team": player_context.get("batter_team", player_context.get("game", "").split(" vs ")[0].strip()),
            "probability": probability,
            "reasoning": reasoning,
            "raw_response": content
        }

    except Exception as e:
        print(f"Error predicting for {player_context.get('player_name', 'unknown')}: {str(e)}")
        return {
            "player": player_context.get("player_name", "Unknown"),
            "team": player_context.get("batter_team", player_context.get("game", "").split(" vs ")[0].strip()),
            "probability": 0.0,
            "reasoning": f"Error: {str(e)}",
            "raw_response": None
        }


def analyze_top_home_run_candidates(top_players: List[Dict[str, Any]], model: str = "llama3") -> Dict[str, Any]:
    """
    Analyzes the top home run candidates and provides a final conclusion.

    Parameters:
        top_players (List[Dict[str, Any]]): List of player predictions with highest probabilities.
        model (str): Name of the locally installed Ollama model to use (e.g., 'llama3').

    Returns:
        Dict with final analysis.
    """
    # Convert JSON to formatted string
    context_str = json.dumps(top_players, indent=2)

    prompt = (
        f"{context_str}\n\n"
        "You are an expert baseball statistician. "
        "The JSON data above contains information about the top players most likely to hit home runs today, "
        "including their calculated probabilities and reasoning. "
        "Analyze this data and provide a final conclusion about which players are most likely to hit home runs today. "
        "Consider the following in your analysis:\n"
        "1. Which player has the strongest case for hitting a home run?\n"
        "2. Are there any patterns or common factors among the top candidates?\n"
        "3. How do ballpark factors influence these predictions?\n"
        "4. Are there any interesting matchups or statistical anomalies worth highlighting?\n\n"
        "Format your response as a detailed analysis with clear sections and conclusions."
    )

    try:
        print("Asking Ollama for final analysis...")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = response['message']['content'].strip()

        return {
            "analysis": content,
            "top_players": top_players
        }

    except Exception as e:
        return {
            "analysis": f"Error generating analysis: {str(e)}",
            "top_players": top_players,
            "error": str(e)
        }