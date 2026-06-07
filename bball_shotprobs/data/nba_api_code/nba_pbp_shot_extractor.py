"""
NBA Play-by-Play Shot Data Extractor
=====================================
Fetches both PlayByPlayV2 and PlayByPlayV3 per game and merges them so that
every shot row contains the full column set from both endpoints.

V2 contributes:  PCTIMESTRING (shot clock), PLAYER2_ID/NAME (assister),
                 PLAYER3_ID/NAME (blocker on misses), EVENTMSGTYPE,
                 EVENTMSGACTIONTYPE, HOMEDESCRIPTION, VISITORDESCRIPTION,
                 SCOREMARGIN, WCTIMESTRING, and all PERSON/PLAYER fields

V3 contributes:  xLegacy, yLegacy, shotDistance, shotResult, isFieldGoal,
                 actionType, subType, location, scoreHome, scoreAway,
                 pointsTotal, clock (period clock), videoAvailable

Merged on:       GAME_ID + EVENTNUM (V2) == GAME_ID + actionNumber (V3)

Additionally, home/away/defensive team columns are derived from
LeagueGameFinder and V3's location field ("H"/"A").

Install:
    pip install nba_api pandas tqdm

Output:
    nba_shots_pbp.csv      — combined CSV across all seasons
"""

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV2, PlayByPlayV3

# ── Configuration ──────────────────────────────────────────────────────────────

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPE = "Regular Season"   # "Regular Season" | "Playoffs" | "All Star"
REQUEST_DELAY = 0.5         # seconds between API calls (per request, not per game)
OUTPUT_DIR = Path(".")

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_game_team_map(season: str, season_type: str) -> tuple[list[str], dict]:
    """
    Returns:
        game_ids  — sorted list of unique game IDs
        team_map  — {game_id: {homeTeamId, homeTeamTricode, awayTeamId, awayTeamTricode}}

    LeagueGameFinder returns two rows per game (one per team).
    MATCHUP format: "BOS vs. MIA" (home) or "BOS @ MIA" (away).
    """
    finder = LeagueGameFinder(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]

    team_map = {}
    for game_id, group in df.groupby("GAME_ID"):
        home_row = group[group["MATCHUP"].str.contains("vs\.", regex=True)]
        away_row = group[group["MATCHUP"].str.contains("@", regex=False)]
        if home_row.empty or away_row.empty:
            continue
        team_map[game_id] = {
            "homeTeamId": home_row.iloc[0]["TEAM_ID"],
            "homeTeamTricode": home_row.iloc[0]["TEAM_ABBREVIATION"],
            "awayTeamId": away_row.iloc[0]["TEAM_ID"],
            "awayTeamTricode": away_row.iloc[0]["TEAM_ABBREVIATION"],
        }

    return sorted(df["GAME_ID"].unique().tolist()), team_map


def fetch_v2(game_id: str) -> pd.DataFrame | None:
    """
    Fetch PlayByPlayV2 and return only shot rows (EVENTMSGTYPE 1=made, 2=missed).
    Renames EVENTNUM → actionNumber for the merge key.
    """
    try:
        df = PlayByPlayV2(
            game_id=game_id,
            start_period=1,
            end_period=10,
        ).get_data_frames()[0]

        if df.empty:
            return None

        shots = df[df["EVENTMSGTYPE"].isin([1, 2])].copy()
        shots = shots.rename(columns={"EVENTNUM": "actionNumber"})
        return shots if not shots.empty else None

    except Exception:
        return None


def fetch_v3(game_id: str) -> pd.DataFrame | None:
    """
    Fetch PlayByPlayV3 and return only shot rows (isFieldGoal == 1).
    """
    try:
        df = PlayByPlayV3(
            game_id=game_id,
            start_period=1,
            end_period=10,
        ).get_data_frames()[0]

        if df.empty:
            return None

        shots = df[df["isFieldGoal"] == 1].copy()
        return shots if not shots.empty else None

    except Exception:
        return None


def add_team_context(shots: pd.DataFrame, game_id: str, team_map: dict) -> pd.DataFrame:
    """
    Add home/away/defensive team columns derived from LeagueGameFinder.
    location == "H" → shooter is home → defender is away, and vice versa.
    """
    info = team_map.get(game_id, {})
    shots["homeTeamId"] = info.get("homeTeamId")
    shots["homeTeamTricode"] = info.get("homeTeamTricode")
    shots["awayTeamId"] = info.get("awayTeamId")
    shots["awayTeamTricode"] = info.get("awayTeamTricode")

    shots["defTeamId"] = shots["location"].map(
        lambda loc: info.get("awayTeamId") if loc == "H"
                    else info.get("homeTeamId") if loc == "A"
                    else None
    )
    shots["defTeamTricode"] = shots["location"].map(
        lambda loc: info.get("awayTeamTricode") if loc == "H"
                    else info.get("homeTeamTricode") if loc == "A"
                    else None
    )
    return shots


def fetch_and_merge(game_id: str, team_map: dict) -> pd.DataFrame | None:
    """
    Fetch V2 and V3 for a game, merge on actionNumber, add team context.
    V2 columns are prefixed with nothing; V3-only columns are kept as-is.
    Duplicate columns (same name, same data) keep the V2 version.
    """
    v2 = fetch_v2(game_id)
    time.sleep(REQUEST_DELAY)
    v3 = fetch_v3(game_id)
    time.sleep(REQUEST_DELAY)

    if v2 is None and v3 is None:
        return None

    # V3-only columns that add value beyond V2
    v3_extra_cols = [
        "actionNumber",   # merge key
        "xLegacy", "yLegacy", "shotDistance", "shotResult",
        "actionType", "subType", "location",
        "actionId",
    ]

    if v2 is not None and v3 is not None:
        v3_subset = v3[[c for c in v3_extra_cols if c in v3.columns]]
        merged = v2.merge(v3_subset, on="actionNumber", how="left")
    elif v2 is not None:
        merged = v2
    else:
        merged = v3

    merged["GAME_ID"] = game_id
    merged = add_team_context(merged, game_id, team_map)
    return merged


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    for season in SEASONS:
        game_ids, team_map = get_game_team_map(season, SEASON_TYPE)
        season_frames: list[pd.DataFrame] = []

        for game_id in tqdm(game_ids, desc=season, unit="game"):
            shots = fetch_and_merge(game_id, team_map)
            if shots is not None:
                shots["SEASON"] = season
                season_frames.append(shots)

        if not season_frames:
            continue

        season_df = pd.concat(season_frames, ignore_index=True)
        all_frames.append(season_df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "nba_shots_pbp.csv", index=False)


if __name__ == "__main__":
    main()
