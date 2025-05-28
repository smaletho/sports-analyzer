import asyncio
from datetime import date, datetime, timezone
from homeruns.mlb_schedule import get_mlb_schedule_async
from homeruns.data_loader import generate_context_summary_async
from homeruns.predictor import predict_single_player_home_run, analyze_top_home_run_candidates
from collections import defaultdict
from operator import itemgetter

async def generate_all_batter_summaries(query_date: str = None) -> list:
    """
    Retrieves all MLB games scheduled today, pulls team data, and generates
    contextual summaries for each batter in the lineup.

    Returns:
        List of strings, each containing a contextual summary for a batter.
    """
    print(f"Starting execution at {datetime.now()}")
    if query_date is not None:
        today_str = query_date
    else:
        today_str = date.today().strftime('%Y-%m-%d')
    schedule = await get_mlb_schedule_async(today_str)

    summaries = []

    for game in schedule['games']:
        game_date_str = game.get('game_date', today_str)
        game_datetime = datetime.strptime(game_date_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)

        now_utc = datetime.now(timezone.utc)
        if now_utc >= game_datetime:
            print(f"Skipping game {game.get('game_id')} — already started.")
            continue  # Game has already started

        lineups = game.get('lineups', {})
        if not lineups:
            print(f"Skipping game {game.get('game_id')} — no lineup data.")
            continue

        # if any(lineups[team]['source'] == 'previous_game' for team in ['home', 'away']):
        #     print(f"Skipping game {game.get('game_id')} — using previous game lineup.")
        #     continue

        away_team = game['away_team']['name']
        home_team = game['home_team']['name']
        ballpark = game.get('venue', {}).get('name', 'Unknown Ballpark')
        team_stats = game.get('team_stats', {})

        # Extract pitcher info
        starting_pitchers = game.get('starting_pitchers', {})
        away_pitcher = starting_pitchers.get('away', {})
        home_pitcher = starting_pitchers.get('home', {})

        # Lineups
        lineups = game.get('lineups', {})
        for side in ['away', 'home']:
            team_name = away_team if side == 'away' else home_team
            opponent_team = home_team if side == 'away' else away_team
            pitcher_info = home_pitcher if side == 'away' else away_pitcher
            try:
                pitcher_name = pitcher_info.get('name')
                pitcher_id = pitcher_info.get('player_id')
            except:
                continue

            if not pitcher_name or not pitcher_id:
                continue  # Skip if no probable pitcher

            lineup_info = lineups.get(side, {})
            players = lineup_info.get('players', [])
            source = lineup_info.get('source', 'unknown')

            for player in players:
                batter_id = player.get('player_id')
                batter_name = player.get('name')

                if not batter_id or not batter_name:
                    continue

                # Generate summary using context
                try:
                    summary = await generate_context_summary_async(
                        batter_id=batter_id,
                        pitcher_id=pitcher_id,
                        batter_team=team_name,
                        pitcher_team=opponent_team,
                        game_date=game_datetime,
                        batter_name=batter_name,
                        pitcher_name=pitcher_name,
                        ballpark_name=ballpark,
                        team_stats=team_stats,
                        source=source
                    )
                    # Add batter_team field to help with team grouping later
                    summary["batter_team"] = team_name
                    summaries.append(summary)

                except Exception as e:
                    print(f"Error generating summary for {batter_name}: {e}")
                    continue
        #break
    #print(f"Finished execution of one matchup at {datetime.now()}")
    return summaries


async def analyze_home_run_probabilities(query_date: str = None, model: str = "llama3") -> dict:
    """
    Analyzes MLB games and predicts home run probabilities for each player,
    then selects top candidates for final analysis.
    
    Args:
        query_date: Optional date string in YYYY-MM-DD format
        model: Name of the Ollama model to use
        
    Returns:
        Dictionary with analysis results
    """
    # Get all player summaries
    player_summaries = await generate_all_batter_summaries(query_date)
    
    if not player_summaries:
        return {"error": "No player summaries generated"}
    
    # Process each player individually
    all_predictions = []
    for summary in player_summaries:
        prediction = predict_single_player_home_run(summary, model)
        all_predictions.append(prediction)
        print(f"Player: {prediction['player']}, Probability: {prediction['probability']}%")
    
    # Group players by team
    teams = defaultdict(list)
    for prediction in all_predictions:
        team = prediction.get("team", "Unknown")
        teams[team].append(prediction)
    
    # Select top 3 players from each team
    top_players = []
    for team, players in teams.items():
        # Sort players by probability (highest first)
        sorted_players = sorted(players, key=itemgetter('probability'), reverse=True)
        # Take top 3 (or fewer if team has less than 3 players)
        top_team_players = sorted_players[:3]
        top_players.extend(top_team_players)
    
    # Get final analysis of top players
    final_analysis = analyze_top_home_run_candidates(top_players, model)
    
    return {
        "all_predictions": all_predictions,
        "top_players": top_players,
        "final_analysis": final_analysis,
        "total_players": len(player_summaries)
    }


# Entry point
if __name__ == '__main__':
    import sys

    # Check if a date parameter was provided
    query_date = None
    if len(sys.argv) > 1:
        query_date = sys.argv[1]
        print(f"Using provided date: {query_date}")

    # Optional model parameter
    model = "llama3"
    if len(sys.argv) > 2:
        model = sys.argv[2]
        print(f"Using model: {model}")

    result = asyncio.run(analyze_home_run_probabilities(query_date, model))
    
    # Print top players
    print("\nTop Home Run Candidates:")
    for player in result.get('top_players', []):
        print(f"Player: {player.get('player')}")
        print(f"Team: {player.get('team')}")
        print(f"Probability: {player.get('probability')}%")
        print(f"Reasoning: {player.get('reasoning')[:200]}...")  # Truncate long reasoning
        print("-" * 40)
    
    # Print final analysis
    print("\nFinal Analysis:")
    print(result.get('final_analysis', {}).get('analysis', 'No analysis available'))
