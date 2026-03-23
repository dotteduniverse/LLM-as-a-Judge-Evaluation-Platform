from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class EvaluationRequest(BaseModel):
    question: str
    candidates: List[str]

class ScoreResponse(BaseModel):
    model: str
    response: str
    factuality: int
    compliance: int
    empathy: int
    score: float

class EvaluationResult(BaseModel):
    results: List[ScoreResponse]

class LeaderboardEntry(BaseModel):
    model: str
    avg_score: float
    avg_factuality: float
    avg_compliance: float
    avg_empathy: float
    num_evaluations: int