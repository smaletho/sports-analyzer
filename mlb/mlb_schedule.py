"""
MLB Schedule Module with Parallel Data Retrieval

This module fetches MLB scheduling information including games, lineups,
starting pitchers, and ballpark information for a given date using parallel requests.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from mlb.data_loader import generate_context_summary, get_team_stats_async, _data_loader


class MLBSchedule:
    """
    A class to fetch MLB scheduling information from the MLB Stats API with parallel processing.
    """

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=30)  # Increased timeout for stability

    async def get_games_for_date(self, date: str) -> Dict[str, Any]:
        """
        Get all MLB games for a specific date with parallel data fetching.
        """
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            schedule_url = f"{self.BASE_URL}/schedule"
            schedule_params = {
                'date': date,
                'sportId': 1,
                'hydrate': 'team,venue,linescore,probablePitcher'
            }

            print("Requesting MLB schedule for date:", date)
            schedule_response = await self._make_request(session, schedule_url, schedule_params)

            if not schedule_response or 'dates' not in schedule_response:
                return {'games': [], 'date': date}

            # Parse games and collect team names
            team_names = set()
            games_raw = []

            for date_info in schedule_response['dates']:
                for game in date_info.get('games', []):
                    games_raw.append(game)
                    away = game.get('teams', {}).get('away', {}).get('team', {}).get('name')
                    home = game.get('teams', {}).get('home', {}).get('team', {}).get('name')
                    if away:
                        team_names.add(away)
                    if home:
                        team_names.add(home)

            print(f"Found {len(games_raw)} games with teams: {list(team_names)}")

            # Fetch team stats using the OptimizedDataLoader
            print("Fetching team statistics in parallel...")
            #team_stats = await _data_loader.get_multiple_team_stats_async(list(team_names), date)
            team_stats = {}
            for team in team_names:
                print(f"Fetching stats for team: {team}")
                team_stats[team] = await get_team_stats_async(team, date)
                print(f"Retrieved stats for team: {team}")
            print(f"Retrieved stats for {len(team_stats)} teams")

            # Process games in parallel
            print("Processing game details...")
            game_tasks = [self._process_game_async(session, g) for g in games_raw]
            game_results = await asyncio.gather(*game_tasks, return_exceptions=True)

            games_data = []
            for i, game_result in enumerate(game_results):
                if isinstance(game_result, Exception):
                    print(f"Error processing game {i+1}: {game_result}")
                    continue
                if game_result:
                    away = game_result.get('away_team', {}).get('name')
                    home = game_result.get('home_team', {}).get('name')

                    # Attach team stats to game data
                    game_result['team_stats'] = {}
                    if away and away in team_stats:
                        game_result['team_stats'][away] = team_stats[away]
                    if home and home in team_stats:
                        game_result['team_stats'][home] = team_stats[home]

                    games_data.append(game_result)

            print(f"Successfully processed {len(games_data)} games")
            return {
                'date': date,
                'games': games_data,
                'team_stats': team_stats
            }

    async def _process_game_async(self, session: aiohttp.ClientSession, game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process individual game data to extract relevant information asynchronously.

        Args:
            session: aiohttp session
            game: Game data from MLB API

        Returns:
            Processed game information
        """
        try:
            game_pk = game.get('gamePk')
            print(f"Processing game {game_pk}")

            # Basic game info
            game_info = {
                'game_id': game_pk,
                'game_date': game.get('gameDate'),
                'status': game.get('status', {}).get('detailedState'),
                'away_team': {
                    'name': game.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                    'abbreviation': game.get('teams', {}).get('away', {}).get('team', {}).get('abbreviation'),
                    'id': game.get('teams', {}).get('away', {}).get('team', {}).get('id')
                },
                'home_team': {
                    'name': game.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                    'abbreviation': game.get('teams', {}).get('home', {}).get('team', {}).get('abbreviation'),
                    'id': game.get('teams', {}).get('home', {}).get('team', {}).get('id')
                },
                'venue': {
                    'name': game.get('venue', {}).get('name'),
                    'id': game.get('venue', {}).get('id')
                }
            }

            # Get starting pitchers
            probable_pitchers = game.get('teams', {})
            if probable_pitchers:
                away_pitcher = probable_pitchers.get('away', {}).get('probablePitcher')
                home_pitcher = probable_pitchers.get('home', {}).get('probablePitcher')

                game_info['starting_pitchers'] = {
                    'away': self._extract_pitcher_info(away_pitcher) if away_pitcher else None,
                    'home': self._extract_pitcher_info(home_pitcher) if home_pitcher else None
                }

            # Get lineups if available
            if game_pk:
                print(f"Fetching lineups for game {game_pk}")
                lineups_data = await self._get_lineups_async(session, game_pk)
                game_info['lineups'] = {
                    'away': {
                        'players': lineups_data['away']['players'],
                        'source': lineups_data['away']['source']
                    },
                    'home': {
                        'players': lineups_data['home']['players'],
                        'source': lineups_data['home']['source']
                    }
                }
                print(f"Found lineups - Away: {len(lineups_data['away']['players'])} players ({lineups_data['away']['source']}), "
                      f"Home: {len(lineups_data['home']['players'])} players ({lineups_data['home']['source']})")

            return game_info

        except Exception as e:
            print(f"Error processing game {game.get('gamePk', 'unknown')}: {e}")
            return None

    def _extract_pitcher_info(self, pitcher_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pitcher information."""
        return {
            'player_id': pitcher_data.get('id'),
            'name': pitcher_data.get('fullName'),
            'hand': pitcher_data.get('pitchHand', {}).get('code')
        }

    async def _get_lineups_async(self, session: aiohttp.ClientSession, game_pk: int) -> Dict[str, Any]:
        """
        Get projected lineups for a specific game asynchronously.

        Args:
            session: aiohttp session
            game_pk: Game primary key

        Returns:
            Dictionary containing home and away lineups with source information
        """
        lineups = {
            'away': {'players': [], 'source': 'none'},
            'home': {'players': [], 'source': 'none'}
        }

        # Try to get lineups from boxscore endpoint (works for games that have started)
        boxscore_url = f"{self.BASE_URL}/game/{game_pk}/boxscore"
        boxscore_response = await self._make_request(session, boxscore_url)

        if boxscore_response and 'teams' in boxscore_response:
            extracted_lineups = self._extract_lineups_from_boxscore(boxscore_response)
            if extracted_lineups['away'] or extracted_lineups['home']:
                lineups['away']['players'] = extracted_lineups['away']
                lineups['home']['players'] = extracted_lineups['home']
                lineups['away']['source'] = 'official' if extracted_lineups['away'] else 'none'
                lineups['home']['source'] = 'official' if extracted_lineups['home'] else 'none'
                print(f"Found official lineups for game {game_pk}")

        # If no lineups found, get team info and try previous game lineups
        if not lineups['away']['players'] and not lineups['home']['players']:
            print(f"No official lineups found for game {game_pk}, trying previous game lineups")
            # Get game info to determine teams
            schedule_url = f"{self.BASE_URL}/schedule"
            schedule_params = {
                'gamePk': game_pk,
                'hydrate': 'team'
            }
            schedule_response = await self._make_request(session, schedule_url, schedule_params)

            if (schedule_response and 'dates' in schedule_response and
                schedule_response['dates']):
                games = schedule_response['dates'][0].get('games', [])
                if games:
                    game_info = games[0]
                    teams = game_info.get('teams', {})

                    away_team_id = teams.get('away', {}).get('team', {}).get('id')
                    home_team_id = teams.get('home', {}).get('team', {}).get('id')

                    # Fetch previous game lineups in parallel
                    tasks = []
                    if away_team_id:
                        tasks.append(self._get_previous_game_lineup_async(session, away_team_id))
                    if home_team_id:
                        tasks.append(self._get_previous_game_lineup_async(session, home_team_id))

                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        result_idx = 0
                        if away_team_id:
                            away_result = results[result_idx]
                            if not isinstance(away_result, Exception) and away_result:
                                lineups['away']['players'] = away_result
                                lineups['away']['source'] = 'previous_game'
                                print(f"Found previous game lineup for away team: {len(away_result)} players")
                            result_idx += 1

                        if home_team_id and result_idx < len(results):
                            home_result = results[result_idx]
                            if not isinstance(home_result, Exception) and home_result:
                                lineups['home']['players'] = home_result
                                lineups['home']['source'] = 'previous_game'
                                print(f"Found previous game lineup for home team: {len(home_result)} players")

        return lineups

    def _extract_lineups_from_boxscore(self, boxscore_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract lineups from boxscore response."""
        lineups = {'away': [], 'home': []}
        teams = boxscore_response.get('teams', {})

        for team_type in ['away', 'home']:
            team_data = teams.get(team_type, {})
            batting_order = team_data.get('battingOrder', [])
            players = team_data.get('players', {})

            lineup = []
            for i, player_id in enumerate(batting_order):
                player_key = f"ID{player_id}"
                if player_key in players:
                    player_info = players[player_key]
                    person = player_info.get('person', {})

                    lineup_entry = {
                        'batting_order': i + 1,
                        'player_id': person.get('id'),
                        'name': person.get('fullName'),
                        'position': player_info.get('position', {}).get('abbreviation'),
                        'bat_side': person.get('batSide', {}).get('code')
                    }
                    lineup.append(lineup_entry)

            lineups[team_type] = lineup

        return lineups

    async def _get_previous_game_lineup_async(self, session: aiohttp.ClientSession, team_id: int) -> List[Dict[str, Any]]:
        """
        Get the starting lineup from the team's most recent completed game asynchronously.

        Args:
            session: aiohttp session
            team_id: Team ID

        Returns:
            List of players from previous game's starting lineup
        """
        current_date = datetime.now()

        # Create tasks for checking multiple dates in parallel (limit to 5 for efficiency)
        date_tasks = []
        for days_back in range(1, 6):
            check_date = (current_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
            date_tasks.append(self._check_date_for_team_game(session, team_id, check_date))

        # Execute date checks in parallel
        results = await asyncio.gather(*date_tasks, return_exceptions=True)

        # Return the first successful lineup found
        for result in results:
            if not isinstance(result, Exception) and result:
                return result

        # If no recent game found, return empty lineup
        return []

    async def _check_date_for_team_game(self, session: aiohttp.ClientSession, team_id: int, check_date: str) -> List[Dict[str, Any]]:
        """
        Check a specific date for completed games for a team.

        Args:
            session: aiohttp session
            team_id: Team ID
            check_date: Date to check in YYYY-MM-DD format

        Returns:
            Lineup if found, empty list otherwise
        """
        # Get schedule for this date
        schedule_url = f"{self.BASE_URL}/schedule"
        schedule_params = {
            'date': check_date,
            'sportId': 1,
            'teamId': team_id
        }

        schedule_response = await self._make_request(session, schedule_url, schedule_params)

        if not schedule_response or 'dates' not in schedule_response:
            return []

        for date_info in schedule_response['dates']:
            for game in date_info.get('games', []):
                # Only look at completed games
                game_state = game.get('status', {}).get('abstractGameState')
                if game_state != 'Final':
                    continue

                game_pk = game.get('gamePk')
                if not game_pk:
                    continue

                # Check if this team played in this game
                teams = game.get('teams', {})
                away_team_id = teams.get('away', {}).get('team', {}).get('id')
                home_team_id = teams.get('home', {}).get('team', {}).get('id')

                if team_id not in [away_team_id, home_team_id]:
                    continue

                # Get boxscore for this completed game
                boxscore_url = f"{self.BASE_URL}/game/{game_pk}/boxscore"
                boxscore_response = await self._make_request(session, boxscore_url)

                if not boxscore_response or 'teams' not in boxscore_response:
                    continue

                # Determine if team was home or away
                boxscore_teams = boxscore_response['teams']
                team_type = None

                if boxscore_teams.get('away', {}).get('team', {}).get('id') == team_id:
                    team_type = 'away'
                elif boxscore_teams.get('home', {}).get('team', {}).get('id') == team_id:
                    team_type = 'home'

                if team_type:
                    lineup = self._extract_team_lineup_from_boxscore(
                        boxscore_response, team_type
                    )
                    if lineup:
                        return lineup

        return []

    def _extract_team_lineup_from_boxscore(self, boxscore_response: Dict[str, Any],
                                         team_type: str) -> List[Dict[str, Any]]:
        """Extract lineup for a specific team from boxscore."""
        teams = boxscore_response.get('teams', {})
        team_data = teams.get(team_type, {})
        batting_order = team_data.get('battingOrder', [])
        players = team_data.get('players', {})

        lineup = []
        for i, player_id in enumerate(batting_order):
            player_key = f"ID{player_id}"
            if player_key in players:
                player_info = players[player_key]
                person = player_info.get('person', {})

                lineup_entry = {
                    'batting_order': i + 1,
                    'player_id': person.get('id'),
                    'name': person.get('fullName'),
                    'position': player_info.get('position', {}).get('abbreviation'),
                    'bat_side': person.get('batSide', {}).get('code')
                }
                lineup.append(lineup_entry)

        return lineup

    async def _make_request(self, session: aiohttp.ClientSession, url: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to MLB API asynchronously with retry logic.

        Args:
            session: aiohttp session
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response data or None if request fails
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        print(f"Rate limited, waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
            except asyncio.TimeoutError:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries}): {url}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

        print(f"All retry attempts failed for: {url}")
        return None


def get_mlb_schedule(date: str) -> Dict[str, Any]:
    """
    Convenience function to get MLB schedule for a specific date with parallel processing.

    Args:
        date (str): Date in YYYY-MM-DD format

    Returns:
        Dictionary containing game information, lineups, and team statistics
    """
    mlb = MLBSchedule()
    return asyncio.run(mlb.get_games_for_date(date))


# Async version for use in async contexts
async def get_mlb_schedule_async(date: str) -> Dict[str, Any]:
    """
    Async version of get_mlb_schedule for use in async contexts.

    Args:
        date (str): Date in YYYY-MM-DD format

    Returns:
        Dictionary containing game information, lineups, and team statistics
    """
    mlb = MLBSchedule()
    return await mlb.get_games_for_date(date)