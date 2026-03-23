import requests
import json
import mlflow
from typing import List, Dict, Any
from src.config import OLLAMA_BASE_URL, JUDGE_MODEL, MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_llm_response(model: str, prompt: str) -> str:
    """Call Ollama to generate a response."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"

def score_response(question: str, candidate_response: str, judge_model: str = JUDGE_MODEL) -> Dict[str, int]:
    """Use the judge model to score the response on factuality, compliance, empathy."""
    prompt = f"""You are an impartial judge evaluating an AI's response to a question.
Rate the response on three metrics (each 1-5):
- Factuality: How accurate is the response? (1 = completely false, 5 = fully correct)
- Compliance: How well does it follow the user's instructions? (1 = completely off, 5 = perfectly aligned)
- Empathy: How appropriate is the tone and understanding? (1 = robotic/offensive, 5 = empathetic/helpful)

Question: {question}
Response: {candidate_response}

Return your ratings in JSON format like this:
{{"factuality": 4, "compliance": 3, "empathy": 5}}
Only output the JSON, no other text.
"""
    judge_output = get_llm_response(judge_model, prompt)
    try:
        # Clean up the output (sometimes contains markdown)
        judge_output = judge_output.strip()
        if judge_output.startswith("```json"):
            judge_output = judge_output.split("```json")[1].split("```")[0]
        elif judge_output.startswith("```"):
            judge_output = judge_output.split("```")[1]
        scores = json.loads(judge_output)
        return scores
    except Exception as e:
        print(f"Error parsing judge output: {e}")
        print("Raw output:", judge_output)
        return {"factuality": 0, "compliance": 0, "empathy": 0}

def evaluate_candidate(question: str, candidate_model: str, judge_model: str = JUDGE_MODEL) -> Dict[str, Any]:
    """Evaluate a single candidate model."""
    response = get_llm_response(candidate_model, question)
    scores = score_response(question, response, judge_model)
    avg_score = sum(scores.values()) / 3
    return {
        "model": candidate_model,
        "response": response,
        "factuality": scores["factuality"],
        "compliance": scores["compliance"],
        "empathy": scores["empathy"],
        "score": avg_score
    }

def evaluate_batch(question: str, candidate_models: List[str], judge_model: str = JUDGE_MODEL) -> List[Dict]:
    results = []
    for model in candidate_models:
        result = evaluate_candidate(question, model, judge_model)
        results.append(result)
        # Log to MLflow
        with mlflow.start_run(run_name=f"eval_{model}_{question[:20]}"):
            mlflow.log_param("question", question)
            mlflow.log_param("model", model)
            mlflow.log_metric("factuality", result["factuality"])
            mlflow.log_metric("compliance", result["compliance"])
            mlflow.log_metric("empathy", result["empathy"])
            mlflow.log_metric("score", result["score"])
    return results