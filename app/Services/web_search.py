import httpx
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


async def search_web(query: str) -> str:
    """Async Tavily web search with timeout."""
    if not TAVILY_API_KEY:
        return "Web search unavailable: TAVILY_API_KEY not set."

    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
        "api_key": TAVILY_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "")
            if answer:
                return answer
            # Fall back to concatenating result snippets
            results = data.get("results", [])
            if results:
                snippets = [r.get("content", "") for r in results[:3]]
                return " ".join(snippets)
            return "No relevant information found via web search."
    except httpx.TimeoutException:
        return "Web search timed out."
    except Exception as e:
        print(f"⚠️  Web search error: {e}")
        return "Web search failed."
