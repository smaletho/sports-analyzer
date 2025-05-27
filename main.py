import asyncio
from datetime import date, datetime
from src.mlb_schedule import get_mlb_schedule_async
from src.data_loader import generate_context_summary_async


async def generate_all_batter_summaries_for_today() -> list:
    """
    Retrieves all MLB games scheduled today, pulls team data, and generates
    contextual summaries for each batter in the lineup.

    Returns:
        List of strings, each containing a contextual summary for a batter.
    """
    print(f"Starting execution at {datetime.now()}")
    today_str = date.today().strftime('%Y-%m-%d')
    schedule = await get_mlb_schedule_async(today_str)

    summaries = []

    for game in schedule['games']:
        game_date_str = game.get('game_date', today_str)
        game_datetime = datetime.strptime(game_date_str, '%Y-%m-%dT%H:%M:%SZ')

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
            pitcher_name = pitcher_info.get('name')
            pitcher_id = pitcher_info.get('player_id')

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
                        team_stats=team_stats
                    )

                    # Mark summary if lineup is from previous game
                    if source == 'previous_game':
                        summary = f"[PREVIOUS GAME LINEUP]\n{summary}"

                    summaries.append(summary)

                except Exception as e:
                    print(f"Error generating summary for {batter_name}: {e}")
                    continue
        break
    print(f"Finished execution of one matchup at {datetime.now()}")
    return summaries


# Entry point
if __name__ == '__main__':
    result = asyncio.run(generate_all_batter_summaries_for_today())
    for summary in result:
        print(summary)
        print("\n" + "=" * 80 + "\n")
