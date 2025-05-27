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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            start_date = (end_date_obj - timedelta(days=days)).strftime('%Y-%m-%d')
        except ValueError:
            end_date_obj = datetime.now()
            start_date = (end_date_obj - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')

        try:
            # Check cache for statcast data
            cache_key = f"statcast_{start_date}_{end_date}"
            if cache_key in _statcast_cache:
                df = _statcast_cache[cache_key]
            else:
                df = statcast(start_date, end_date)
                _statcast_cache[cache_key] = df

            # Filter for the specific team
            df_team = df[(df['home_team'] == team) | (df['away_team'] == team)]

            if df_team.empty:
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
    return playerid_lookup(first_name, last_name)


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
            df = statcast_batter(start_date, end_date_str, batter_id)
            _statcast_cache[cache_key] = df

        if df.empty:
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
            df = statcast_pitcher(start_date, end_date_str, pitcher_id)
            _statcast_cache[cache_key] = df

        if df.empty:
            return {
                "summary": "No recent pitcher data available.",
                "hr_allowed": 0,
                "total_batters": 0,
                "avg_ev": 0
            }

        hr_allowed = (df['events'].fillna('') == 'home_run').sum()
        total_batters = len(df)
        avg_ev = df['launch_speed'].mean()

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

    cache_key = f"batter_full_{batter_id}_{start_date}_{end_date}"

    try:
        if cache_key in _statcast_cache:
            full_df = _statcast_cache[cache_key]
        else:
            full_df = statcast_batter(start_date, end_date, batter_id)
            _statcast_cache[cache_key] = full_df

        if full_df.empty:
            return "No historical batter data available.", "No historical team data available."

        # Vectorized team assignment
        full_df['batter_team'] = full_df.apply(
            lambda row: row['away_team'] if row['inning_topbot'] == 'Top' else row['home_team'],
            axis=1
        )
        full_df['opponent_team'] = full_df.apply(
            lambda row: row['home_team'] if row['inning_topbot'] == 'Top' else row['away_team'],
            axis=1
        )

        vs_pitcher = full_df[full_df['pitcher'] == pitcher_id]
        vs_team = full_df[full_df['opponent_team'] == opponent_team]

        def summarize_optimized(df: pd.DataFrame, label: str) -> str:
            if df.empty:
                return f"No {label} data available."

            hr = (df['events'].fillna('') == 'home_run').sum()
            ab = len(df)
            avg_ev = df['launch_speed'].mean()

            return f"{label}: {hr} HRs in {ab} ABs. Avg EV: {avg_ev:.1f} mph."

        return summarize_optimized(vs_pitcher, "Vs Pitcher"), summarize_optimized(vs_team, "Vs Team")

    except Exception as e:
        logger.error(f"Error in matchup stats: {e}")
        return f"Error in pitcher matchup: {str(e)}", f"Error in team matchup: {str(e)}"


@lru_cache(maxsize=500)
def get_hand_splits_cached(batter_name: str, pitcher_name: str) -> str:
    """
    Cached version of handedness lookup.
    """
    try:
        batter_parts = batter_name.split()
        pitcher_parts = pitcher_name.split()

        if len(batter_parts) < 2 or len(pitcher_parts) < 2:
            return "Handedness info not available - invalid names."

        batter = get_cached_player_lookup(batter_parts[0], batter_parts[1])
        pitcher = get_cached_player_lookup(pitcher_parts[0], pitcher_parts[1])

        if batter.empty or pitcher.empty:
            return "Handedness info not available."

        batter_hand = batter.iloc[0]['bats']
        pitcher_hand = pitcher.iloc[0]['throws']

        if pd.isna(batter_hand) or pd.isna(pitcher_hand):
            return "Handedness data missing."

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


async def generate_context_summary_async(batter_id: int, pitcher_id: int, batter_team: str,
                                         pitcher_team: str, game_date: datetime,
                                         batter_name: str, pitcher_name: str,
                                         ballpark_name: str, team_stats: dict = None) -> str:
    """
    Async version of generate_context_summary with parallel data fetching.
    """
    # Create tasks for parallel execution
    tasks = [
        _data_loader._get_batter_stats_async(batter_id, game_date),
        _data_loader._get_pitcher_stats_async(pitcher_id, game_date)
    ]

    # Execute batter and pitcher stats in parallel
    batter_result, pitcher_result = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle results
    recent_batter = batter_result.get("summary", "Error") if not isinstance(batter_result,
                                                                            Exception) else "Error retrieving batter stats"
    pitcher_stats = pitcher_result.get("summary", "Error") if not isinstance(pitcher_result,
                                                                             Exception) else "Error retrieving pitcher stats"

    # Get matchup stats (run in thread pool since it's CPU intensive)
    loop = asyncio.get_event_loop()
    vs_pitcher, vs_team = await loop.run_in_executor(
        _data_loader.thread_executor,
        get_matchup_stats_optimized,
        batter_id, pitcher_id, pitcher_team, game_date
    )

    # Use pre-computed team stats if available
    if team_stats and batter_team in team_stats:
        batter_team_form = team_stats[batter_team]["summary"]
    else:
        batter_team_stats = await get_team_stats_async(batter_team, game_date.strftime('%Y-%m-%d'))
        batter_team_form = batter_team_stats["summary"]

    if team_stats and pitcher_team in team_stats:
        pitcher_team_form = team_stats[pitcher_team]["summary"]
    else:
        pitcher_team_stats = await get_team_stats_async(pitcher_team, game_date.strftime('%Y-%m-%d'))
        pitcher_team_form = pitcher_team_stats["summary"]

    # Get cached handedness and ballpark info
    handedness = get_hand_splits_cached(batter_name, pitcher_name)
    ballpark_factor = get_ballpark_factor_cached(ballpark_name)

    return "\n".join([
        f"=== {batter_name} vs {pitcher_name} on {game_date.strftime('%Y-%m-%d')} ===",
        "",
        "=== Batter Recent Performance ===",
        recent_batter,
        "",
        "=== Matchup History ===",
        vs_pitcher,
        vs_team,
        "",
        "=== Team Trends ===",
        batter_team_form,
        pitcher_team_form,
        "",
        "=== Pitcher Recent Performance ===",
        pitcher_stats,
        "",
        "=== Handedness Matchup ===",
        handedness,
        "",
        "=== Park Factors ===",
        ballpark_factor
    ])


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
    from src.mlb_schedule import get_mlb_schedule_async

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