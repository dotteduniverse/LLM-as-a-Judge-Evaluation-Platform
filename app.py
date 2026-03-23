import streamlit as st
import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="LLM Evaluation Platform", page_icon="📊")
st.title("📊 LLM Evaluation Platform")

tab1, tab2, tab3 = st.tabs(["Leaderboard", "Evaluate", "MLflow"])

with tab1:
    st.header("Leaderboard")
    response = requests.get(f"{API_URL}/leaderboard")
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["leaderboard"])
        st.dataframe(df)
    else:
        st.error("Could not load leaderboard")

with tab2:
    st.header("Evaluate a Question")
    question = st.text_input("Question:")
    candidates = st.multiselect("Candidate Models", ["llama3.2", "mistral", "phi3"], default=["llama3.2"])
    if st.button("Evaluate"):
        if question and candidates:
            payload = {"question": question, "candidates": candidates}
            response = requests.post(f"{API_URL}/evaluate", json=payload)
            if response.status_code == 200:
                results = response.json()
                st.write("Results:")
                for r in results:
                    st.write(f"**{r['model']}**: Score {r['score']:.2f}")
                    st.write(f"  - Factuality: {r['factuality']}, Compliance: {r['compliance']}, Empathy: {r['empathy']}")
                    st.write(f"  - Response: {r['response'][:200]}...")
            else:
                st.error("Evaluation failed")

with tab3:
    st.header("MLflow Tracking")
    st.markdown("View runs at [MLflow UI](http://localhost:5000)")