import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import logging
from typing import Dict, List, Optional, Any, Tuple
from pybaseball import (
    statcast_batter,
    playerid_lookup,
    statcast,
    statcast_pitcher,
    batting_stats_range,
    pitching_stats_range
)
import os
import json


import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from MLB API team names to pybaseball team abbreviations
MLB_TEAM_MAPPING = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN"
}

# Approximate home run factor by ballpark (100 = neutral)
BALLPARK_HR_FACTORS = {
    "Angel Stadium": 104,
    "American Family Field": 106,
    "Busch Stadium": 92,
    "Chase Field": 98,
    "Citi Field": 107,
    "Citizens Bank Park": 112,
    "Comerica Park": 93,
    "Coors Field": 120,
    "Dodger Stadium": 102,
    "Fenway Park": 105,
    "Globe Life Field": 110,
    "Great American Ball Park": 114,
    "Guaranteed Rate Field": 103,
    "Kauffman Stadium": 90,
    "loanDepot Park": 95,
    "Minute Maid Park": 101,
    "Nationals Park": 105,
    "Oakland Coliseum": 89,
    "Oracle Park": 85,
    "Oriole Park at Camden Yards": 98,
    "PNC Park": 94,
    "Petco Park": 96,
    "Progressive Field": 100,
    "Rogers Centre": 104,
    "T-Mobile Park": 97,
    "Target Field": 99,
    "Tropicana Field": 90,
    "Truist Park": 102,
    "Wrigley Field": 108,
    "Yankee Stadium": 115,
}

# Global cache for frequently accessed data
_player_lookup_cache = {}
_statcast_cache = {}
_team_stats_cache = {}


class OptimizedDataLoader:
    """
    Optimized data loader with parallel processing capabilities.
    """

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, max_workers))

    async def get_multiple_team_stats_async(self, teams: List[str], end_date: str, days: int = 14) -> Dict[str, Any]:
        """
        Get team statistics for multiple teams in parallel.

        Args:
            teams: List of team names
            end_date: End date in YYYY-MM-DD format
            days: Number of days to look back

        Returns:
            Dictionary mapping team names to their statistics
        """
        cached = load_team_stats_from_cache(end_date)
        if cached:
            logger.info(f"Loaded team stats from cache for {end_date}")
            return cached

        # Create tasks for all teams
        tasks = [
            self._get_team_stats_optimized_async(team, end_date, days)
            for team in teams
        ]

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        team_stats = {}
        for team, result in zip(teams, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching stats for {team}: {result}")
                team_stats[team] = {
                    "team": team,
                    "error": str(result),
                    "summary": f"Error retrieving stats for {team}"
                }
            else:
                team_stats[team] = result

        # Save to disk for reuse
        save_team_stats_to_cache(team_stats, end_date)
        return team_stats

    async def _get_team_stats_optimized_async(self, team: str, end_date: str, days: int = 14) -> Dict[str, Any]:
        """
        Optimized async version of get_team_stats with caching.
        """
        cache_key = f"{team}_{end_date}_{days}"

        # Check cache first
        if cache_key in _team_stats_cache:
            return _team_stats_cache[cache_key]

        # Run the computation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_executor,
            self._get_team_stats_sync,
            team, end_date, days
        )

        # Cache the result
        _team_stats_cache[cache_key] = result
        return result

    def _get_team_stats_sync(self, team: str, end_date: str, days: int = 14) -> Dict[str, Any]:
        """
        Synchronous team stats calculation optimized for parallel execution.
        """
        try:
            # Convert team name to abbreviation
            team_abbrev = MLB_TEAM_MAPPING.get(team, team)

            # Parse end date
            try:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                end_date_obj = datetime.now()
                end_date = end_date_obj.strftime('%Y-%m-%d')

            start_date = (end_date_obj - timedelta(days=days)).strftime('%Y-%m-%d')

            # Check cache for statcast data
            cache_key = f"statcast_{start_date}_{end_date}"
            if cache_key in _statcast_cache:
                df = _statcast_cache[cache_key]
            else:
                logger.info(f"Fetching Statcast data from {start_date} to {end_date}")
                df = statcast(start_date, end_date)
                _statcast_cache[cache_key] = df

            if df.empty:
                logger.warning(f"No Statcast data available for date range {start_date} to {end_date}")
                return {
                    "team": team,
                    "total_hr": 0,
                    "total_hits": 0,
                    "avg_ev": 0,
                    "days": days,
                    "summary": f"No recent data for team {team}."
                }

            # Filter for the specific team using abbreviation
            df_team = df[(df['home_team'] == team_abbrev) | (df['away_team'] == team_abbrev)]

            if df_team.empty:
                logger.warning(f"No data found for team {team} (abbrev: {team_abbrev})")
                return {
                    "team": team,
                    "total_hr": 0,
                    "total_hits": 0,
                    "avg_ev": 0,
                    "days": days,
                    "summary": f"No recent data for team {team}."
                }

            # Vectorized calculations for better performance
            hr_mask = df_team['events'] == 'home_run'
            hits_mask = df_team['events'].isin(['single', 'double', 'triple', 'home_run'])

            total_hr = hr_mask.sum()
            total_hits = hits_mask.sum()
            avg_ev = df_team['launch_speed'].mean()

            logger.info(f"Team stats for {team}: {total_hr} HRs, {total_hits} hits")

            return {
                "team": team,
                "total_hr": int(total_hr),
                "total_hits": int(total_hits),
                "avg_ev": float(avg_ev) if not pd.isna(avg_ev) else 0,
                "days": days,
                "summary": f"{team} hit {total_hr} HRs in last {days} days with avg EV of {avg_ev:.1f} mph."
            }
        except Exception as e:
            logger.error(f"Error in team stats for {team}: {e}")
            return {
                "team": team,
                "error": str(e),
                "summary": f"Error retrieving stats for {team}: {str(e)}"
            }

    async def get_multiple_player_stats_async(self, player_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get statistics for multiple players in parallel.

        Args:
            player_requests: List of dictionaries containing player request parameters

        Returns:
            List of player statistics results
        """
        tasks = []
        for request in player_requests:
            if request['type'] == 'batter':
                task = self._get_batter_stats_async(
                    request['player_id'],
                    request['end_date'],
                    request.get('days', 14)
                )
            elif request['type'] == 'pitcher':
                task = self._get_pitcher_stats_async(
                    request['player_id'],
                    request['end_date'],
                    request.get('days', 30)
                )
            else:
                continue
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]

    async def _get_batter_stats_async(self, batter_id: int, end_date: datetime, days: int = 14) -> Dict[str, Any]:
        """Async wrapper for batter stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_executor,
            get_recent_batter_stats_optimized,
            batter_id, end_date, days
        )

    async def _get_pitcher_stats_async(self, pitcher_id: int, end_date: datetime, days: int = 30) -> Dict[str, Any]:
        """Async wrapper for pitcher stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_executor,
            get_pitcher_stats_optimized,
            pitcher_id, end_date, days
        )

    def __del__(self):
        """Cleanup executors."""
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=False)
        if hasattr(self, 'process_executor'):
            self.process_executor.shutdown(wait=False)


# Global optimized data loader instance
_data_loader = OptimizedDataLoader()


@lru_cache(maxsize=1000)
def get_cached_player_lookup(first_name: str, last_name: str) -> pd.DataFrame:
    """Cached player lookup to avoid repeated API calls."""
    try:
        logger.info(f"Looking up player: {first_name} {last_name}")
        result = playerid_lookup(first_name, last_name)
        logger.info(f"Player lookup result shape: {result.shape if not result.empty else 'empty'}")
        return result
    except Exception as e:
        logger.error(f"Error in player lookup for {first_name} {last_name}: {e}")
        return pd.DataFrame()


def get_recent_batter_stats_optimized(batter_id: int, end_date: datetime, days: int = 14) -> Dict[str, Any]:
    """
    Optimized version of get_recent_batter_stats with better error handling and caching.
    """
    start_date = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    cache_key = f"batter_{batter_id}_{start_date}_{end_date_str}"

    try:
        # Check cache first
        if cache_key in _statcast_cache:
            df = _statcast_cache[cache_key]
        else:
            logger.info(f"Fetching batter stats for ID {batter_id} from {start_date} to {end_date_str}")
            df = statcast_batter(start_date, end_date_str, batter_id)
            _statcast_cache[cache_key] = df

        if df.empty:
            logger.warning(f"No batter data for ID {batter_id}")
            return {
                "summary": "No recent batter data available.",
                "hr": 0,
                "ab": 0,
                "avg_ev": 0
            }

        # Vectorized operations
        hr = (df['events'].fillna('') == 'home_run').sum()
        ab = len(df)
        avg_ev = df['launch_speed'].mean()

        logger.info(f"Batter {batter_id} stats: {hr} HRs in {ab} ABs")

        return {
            "summary": f"Recent: {hr} HRs in {ab} ABs. Avg EV: {avg_ev:.1f} mph.",
            "hr": int(hr),
            "ab": ab,
            "avg_ev": float(avg_ev) if not pd.isna(avg_ev) else 0
        }
    except Exception as e:
        logger.error(f"Error in batter stats for {batter_id}: {e}")
        return {
            "summary": f"Error retrieving batter data: {str(e)}",
            "error": str(e)
        }


def get_pitcher_stats_optimized(pitcher_id: int, end_date: datetime, days: int = 30) -> Dict[str, Any]:
    """
    Optimized version of get_pitcher_stats with caching.
    """
    start_date = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    cache_key = f"pitcher_{pitcher_id}_{start_date}_{end_date_str}"

    try:
        if cache_key in _statcast_cache:
            df = _statcast_cache[cache_key]
        else:
            logger.info(f"Fetching pitcher stats for ID {pitcher_id} from {start_date} to {end_date_str}")
            df = statcast_pitcher(start_date, end_date_str, pitcher_id)
            _statcast_cache[cache_key] = df

        if df.empty:
            logger.warning(f"No pitcher data for ID {pitcher_id}")
            return {
                "summary": "No recent pitcher data available.",
                "hr_allowed": 0,
                "total_batters": 0,
                "avg_ev": 0
            }

        hr_allowed = (df['events'].fillna('') == 'home_run').sum()
        total_batters = len(df)
        avg_ev = df['launch_speed'].mean()

        logger.info(f"Pitcher {pitcher_id} stats: {hr_allowed} HRs allowed in {total_batters} batters")

        return {
            "summary": f"Pitcher allowed {hr_allowed} HRs in {total_batters} batters. Avg EV allowed: {avg_ev:.1f} mph.",
            "hr_allowed": int(hr_allowed),
            "total_batters": total_batters,
            "avg_ev": float(avg_ev) if not pd.isna(avg_ev) else 0
        }
    except Exception as e:
        logger.error(f"Error in pitcher stats for {pitcher_id}: {e}")
        return {
            "summary": f"Error retrieving pitcher data: {str(e)}",
            "error": str(e)
        }


def get_matchup_stats_optimized(batter_id: int, pitcher_id: int, opponent_team: str, game_date: datetime) -> Tuple[
    str, str]:
    """
    Optimized version of get_matchup_stats with better data processing.
    """
    start_date = "2018-01-01"
    end_date = game_date.strftime('%Y-%m-%d')

    # Convert opponent team to abbreviation
    opponent_abbrev = MLB_TEAM_MAPPING.get(opponent_team, opponent_team)

    cache_key = f"batter_full_{batter_id}_{start_date}_{end_date}"

    try:
        if cache_key in _statcast_cache:
            full_df = _statcast_cache[cache_key]
        else:
            logger.info(f"Fetching full batter history for ID {batter_id}")
            full_df = statcast_batter(start_date, end_date, batter_id)
            _statcast_cache[cache_key] = full_df

        if full_df.empty:
            logger.warning(f"No historical batter data for ID {batter_id}")
            return "No historical batter data available.", "No historical team data available."

        # Filter matchups
        vs_pitcher = full_df[full_df['pitcher'] == pitcher_id]

        # For team matchups, we need to determine which team the batter was facing
        # If batter's team was away (top inning), they faced the home team
        # If batter's team was home (bottom inning), they faced the away team
        full_df['opponent'] = full_df.apply(
            lambda row: row['home_team'] if row['inning_topbot'] == 'Top' else row['away_team'],
            axis=1
        )
        vs_team = full_df[full_df['opponent'] == opponent_abbrev]

        def summarize_optimized(df: pd.DataFrame, label: str) -> str:
            if df.empty:
                return f"No {label} data available."

            hr = (df['events'].fillna('') == 'home_run').sum()
            ab = len(df)
            avg_ev = df['launch_speed'].mean()

            return f"{label}: {hr} HRs in {ab} ABs. Avg EV: {avg_ev:.1f} mph."

        vs_pitcher_summary = summarize_optimized(vs_pitcher, "Vs Pitcher")
        vs_team_summary = summarize_optimized(vs_team, "Vs Team")

        logger.info(f"Matchup stats - vs pitcher: {len(vs_pitcher)} ABs, vs team: {len(vs_team)} ABs")

        return vs_pitcher_summary, vs_team_summary

    except Exception as e:
        logger.error(f"Error in matchup stats: {e}")
        return f"Error in pitcher matchup: {str(e)}", f"Error in team matchup: {str(e)}"


@lru_cache(maxsize=500)
def get_hand_splits_cached(batter_name: str, pitcher_name: str) -> str:
    """
    Cached version of handedness lookup with improved name parsing.
    """
    try:
        # Clean and parse names
        batter_name = batter_name.strip()
        pitcher_name = pitcher_name.strip()

        # Handle names with suffixes (Jr., Sr., III, etc.)
        def parse_name(full_name):
            parts = full_name.split()
            if len(parts) < 2:
                return None, None

            # Remove common suffixes
            suffixes = ['Jr.', 'Sr.', 'II', 'III', 'IV', 'V']
            while parts and parts[-1] in suffixes:
                parts.pop()

            if len(parts) < 2:
                return None, None

            first_name = parts[0]
            last_name = ' '.join(parts[1:])  # Handle multi-part last names
            return first_name, last_name

        batter_first, batter_last = parse_name(batter_name)
        pitcher_first, pitcher_last = parse_name(pitcher_name)

        if not all([batter_first, batter_last, pitcher_first, pitcher_last]):
            logger.warning(f"Could not parse names: '{batter_name}', '{pitcher_name}'")
            return "Handedness info not available - invalid names."

        logger.info(f"Looking up handedness: {batter_first} {batter_last} vs {pitcher_first} {pitcher_last}")

        batter = get_cached_player_lookup(batter_first, batter_last)
        pitcher = get_cached_player_lookup(pitcher_first, pitcher_last)

        if batter.empty or pitcher.empty:
            logger.warning(f"Player lookup failed - batter empty: {batter.empty}, pitcher empty: {pitcher.empty}")
            return "Handedness info not available."

        batter_hand = batter.iloc[0]['bats']
        pitcher_hand = pitcher.iloc[0]['throws']

        if pd.isna(batter_hand) or pd.isna(pitcher_hand):
            logger.warning(f"Handedness data missing - batter: {batter_hand}, pitcher: {pitcher_hand}")
            return "Handedness data missing."

        logger.info(f"Handedness found - batter bats: {batter_hand}, pitcher throws: {pitcher_hand}")
        return f"Batter bats {batter_hand}, Pitcher throws {pitcher_hand}. Matchup: {batter_hand} vs {pitcher_hand}"

    except Exception as e:
        logger.error(f"Error in handedness lookup: {e}")
        return f"Error retrieving handedness info: {str(e)}"


@lru_cache(maxsize=100)
def get_ballpark_factor_cached(ballpark_name: str) -> str:
    """Cached ballpark factor lookup."""
    factor = BALLPARK_HR_FACTORS.get(ballpark_name, 100)
    return f"Ballpark factor: {factor} (100 = avg)."


async def get_team_stats_async(team: str, end_date: str, days: int = 14) -> Dict[str, Any]:
    """
    Async wrapper for get_team_stats using the global optimized data loader.
    """
    return await _data_loader._get_team_stats_optimized_async(team, end_date, days)


def get_team_stats(team: str, end_date: str, days: int = 14) -> Dict[str, Any]:
    """
    Synchronous version that uses async internally for compatibility.
    """
    return asyncio.run(get_team_stats_async(team, end_date, days))


async def generate_context_summary_async(
    batter_id: int,
    pitcher_id: int,
    batter_team: str,
    pitcher_team: str,
    game_date: datetime,
    batter_name: str,
    pitcher_name: str,
    ballpark_name: str,
    team_stats: dict = None,
    source: str = "previous_game"
) -> dict:
    """
    Async version of generate_context_summary that returns structured JSON data.
    """
    # Create tasks for parallel execution
    tasks = [
        _data_loader._get_batter_stats_async(batter_id, game_date),
        _data_loader._get_pitcher_stats_async(pitcher_id, game_date)
    ]

    # Execute batter and pitcher stats in parallel
    batter_result, pitcher_result = await asyncio.gather(*tasks, return_exceptions=True)

    # Parse results
    batter_stats = batter_result if not isinstance(batter_result, Exception) else {}
    pitcher_stats_data = pitcher_result if not isinstance(pitcher_result, Exception) else {}

    recent_batter = batter_stats.get("parsed", {})  # Use parsed version instead of text summary
    recent_pitcher = pitcher_stats_data.get("parsed", {})

    # Get matchup stats (CPU intensive)
    loop = asyncio.get_event_loop()
    vs_pitcher_stats, vs_team_stats = await loop.run_in_executor(
        _data_loader.thread_executor,
        get_matchup_stats_optimized,
        batter_id, pitcher_id, pitcher_team, game_date
    )

    # Get team stats (batting and pitching)
    if team_stats and batter_team in team_stats:
        batter_team_form = team_stats[batter_team].get("parsed", {})
    else:
        batter_team_stats = await get_team_stats_async(batter_team, game_date.strftime('%Y-%m-%d'))
        batter_team_form = batter_team_stats.get("parsed", {})

    if team_stats and pitcher_team in team_stats:
        pitcher_team_form = team_stats[pitcher_team].get("parsed", {})
    else:
        pitcher_team_stats = await get_team_stats_async(pitcher_team, game_date.strftime('%Y-%m-%d'))
        pitcher_team_form = pitcher_team_stats.get("parsed", {})

    # Get handedness and ballpark factor
    handedness_info = get_hand_splits_cached(batter_name, pitcher_name)
    park_factor = get_ballpark_factor_cached(ballpark_name)

    # Final JSON object
    return {
        "player_name": batter_name,
        "game_date": game_date.strftime('%Y-%m-%d'),
        "game": f"{batter_team} vs {pitcher_team}",
        "opposing_pitcher": pitcher_name,
        "ballpark": ballpark_name,
        "ballpark_factor": park_factor,
        "handedness_matchup": handedness_info,
        "batter_recent_performance": recent_batter,
        "matchup_history_vs_pitcher": vs_pitcher_stats,
        "matchup_history_vs_team": vs_team_stats,
        "batter_team_trend": batter_team_form,
        "pitcher_team_trend": pitcher_team_form,
        "pitcher_recent_performance": recent_pitcher,
        "source": source
    }



def generate_context_summary(batter_id: int, pitcher_id: int, batter_team: str,
                             pitcher_team: str, game_date: datetime,
                             batter_name: str, pitcher_name: str,
                             ballpark_name: str, team_stats: dict = None) -> str:
    """
    Synchronous wrapper for generate_context_summary_async.
    """
    return asyncio.run(generate_context_summary_async(
        batter_id, pitcher_id, batter_team, pitcher_team, game_date,
        batter_name, pitcher_name, ballpark_name, team_stats
    ))


async def fetch_daily_data_async() -> Dict[str, Any]:
    """
    Async version of fetch_daily_data with optimized parallel processing.
    """
    from datetime import date
    today = date.today().strftime('%Y-%m-%d')

    # Import the async version
    from mlb.mlb_schedule import get_mlb_schedule_async

    # Get today's schedule with parallel processing
    schedule = await get_mlb_schedule_async(today)

    return schedule


def fetch_daily_data() -> Dict[str, Any]:
    """
    Synchronous wrapper for fetch_daily_data_async.
    """
    return asyncio.run(fetch_daily_data_async())


# Legacy function compatibility
def get_recent_batter_stats(batter_id: int, end_date: datetime, days: int = 14) -> str:
    """Legacy compatibility function."""
    result = get_recent_batter_stats_optimized(batter_id, end_date, days)
    return result.get("summary", "No data available")


def get_matchup_stats(batter_id: int, pitcher_id: int, opponent_team: str, game_date: datetime) -> Tuple[str, str]:
    """Legacy compatibility function."""
    return get_matchup_stats_optimized(batter_id, pitcher_id, opponent_team, game_date)


def get_team_form(team: str, end_date: datetime, days: int = 14) -> str:
    """Legacy compatibility function."""
    stats = get_team_stats(team, end_date.strftime('%Y-%m-%d'), days)
    return stats.get("summary", f"No recent data for team {team}.")


def get_pitcher_stats(pitcher_id: int, end_date: datetime, days: int = 30) -> str:
    """Legacy compatibility function."""
    result = get_pitcher_stats_optimized(pitcher_id, end_date, days)
    return result.get("summary", "No data available")


def get_hand_splits(batter_name: str, pitcher_name: str) -> str:
    """Legacy compatibility function."""
    return get_hand_splits_cached(batter_name, pitcher_name)


def get_ballpark_factor(ballpark_name: str) -> str:
    """Legacy compatibility function."""
    return get_ballpark_factor_cached(ballpark_name)


# Cache management functions
def clear_caches():
    """Clear all caches to free memory."""
    global _player_lookup_cache, _statcast_cache, _team_stats_cache
    _player_lookup_cache.clear()
    _statcast_cache.clear()
    _team_stats_cache.clear()

    # Clear LRU caches
    get_cached_player_lookup.cache_clear()
    get_hand_splits_cached.cache_clear()
    get_ballpark_factor_cached.cache_clear()


def get_cache_info() -> Dict[str, Any]:
    """Get information about current cache usage."""
    return {
        "player_lookup_cache": get_cached_player_lookup.cache_info(),
        "handedness_cache": get_hand_splits_cached.cache_info(),
        "ballpark_cache": get_ballpark_factor_cached.cache_info(),
        "statcast_cache_size": len(_statcast_cache),
        "team_stats_cache_size": len(_team_stats_cache)
    }

def save_team_stats_to_cache(stats: Dict[str, Any], date_str: str) -> None:
    os.makedirs("cache", exist_ok=True)
    with open(f"cache/team_stats_{date_str}.json", "w") as f:
        json.dump(stats, f)


def load_team_stats_from_cache(date_str: str) -> Optional[Dict[str, Any]]:
    path = f"cache/team_stats_{date_str}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
