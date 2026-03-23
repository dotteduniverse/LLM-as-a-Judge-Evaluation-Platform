import json
import os
from typing import List, Dict
from src.models import LeaderboardEntry

LEADERBOARD_FILE = "data/leaderboard.json"

def load_leaderboard() -> List[LeaderboardEntry]:
    if not os.path.exists(LEADERBOARD_FILE):
        return []
    with open(LEADERBOARD_FILE, "r") as f:
        data = json.load(f)
        return [LeaderboardEntry(**entry) for entry in data]

def save_leaderboard(entries: List[LeaderboardEntry]):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump([entry.dict() for entry in entries], f, indent=2)

def update_leaderboard(new_result: Dict):
    entries = load_leaderboard()
    # Find if model already exists
    existing = next((e for e in entries if e.model == new_result["model"]), None)
    if existing:
        n = existing.num_evaluations
        existing.avg_factuality = (existing.avg_factuality * n + new_result["factuality"]) / (n + 1)
        existing.avg_compliance = (existing.avg_compliance * n + new_result["compliance"]) / (n + 1)
        existing.avg_empathy = (existing.avg_empathy * n + new_result["empathy"]) / (n + 1)
        existing.avg_score = (existing.avg_score * n + new_result["score"]) / (n + 1)
        existing.num_evaluations += 1
    else:
        entries.append(LeaderboardEntry(
            model=new_result["model"],
            avg_score=new_result["score"],
            avg_factuality=new_result["factuality"],
            avg_compliance=new_result["compliance"],
            avg_empathy=new_result["empathy"],
            num_evaluations=1
        ))
    save_leaderboard(entries)