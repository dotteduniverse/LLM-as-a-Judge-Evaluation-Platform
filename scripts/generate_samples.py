"""
Generate synthetic Q&A pairs using Ollama.
"""
import requests
import json
import os
from src.config import OLLAMA_BASE_URL, JUDGE_MODEL

def generate_qa(model, num_samples=10):
    samples = []
    for i in range(num_samples):
        prompt = f"Generate a random question and its correct answer about a general topic. Return in JSON: {{'question': ..., 'answer': ...}}"
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": model, "prompt": prompt, "stream": False})
        if response.status_code == 200:
            text = response.json()["response"]
            try:
                # Extract JSON
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    obj = json.loads(text[start:end])
                    samples.append(obj)
                else:
                    print(f"Failed to parse response {i}: {text}")
            except Exception as e:
                print(f"Error parsing: {e}")
                continue
    return samples

if __name__ == "__main__":
    samples = generate_qa(JUDGE_MODEL, num_samples=10)
    os.makedirs("data", exist_ok=True)
    with open("data/synthetic_qa.json", "w") as f:
        json.dump(samples, f, indent=2)
    print("Saved synthetic Q&A to data/synthetic_qa.json")