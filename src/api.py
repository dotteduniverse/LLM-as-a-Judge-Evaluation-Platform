from fastapi import FastAPI, HTTPException
from src.models import EvaluationRequest, EvaluationResult, LeaderboardEntry
from src.evaluator import evaluate_batch
from src.leaderboard import load_leaderboard, update_leaderboard
from typing import List

app = FastAPI(title="LLM Evaluation Platform")

@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate(request: EvaluationRequest):
    try:
        results = evaluate_batch(request.question, request.candidates)
        # Update leaderboard with each result
        for r in results:
            update_leaderboard(r)
        return EvaluationResult(results=r)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
async def get_leaderboard():
    entries = load_leaderboard()
    # Sort by avg_score descending
    entries.sort(key=lambda x: x.avg_score, reverse=True)
    return {"leaderboard": [entry.dict() for entry in entries]}

@app.get("/health")
async def health():
    return {"status": "ok"}