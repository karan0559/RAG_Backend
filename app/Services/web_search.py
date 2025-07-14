# Services/web_search.py

import requests
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def search_web(query: str) -> str:
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("answer", "No relevant info found via web search.")
