import httpx
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Note: keys with the "tvly-dev-" prefix are trial/dev tier and may have
# stricter rate limits or expiry.  Replace with a production key if you hit
# quota errors.  See https://tavily.com for account management.


async def search_web(query: str) -> str:
    """Async Tavily web search with timeout and graceful error handling."""
    if not TAVILY_API_KEY:
        return "Web search unavailable: TAVILY_API_KEY is not set in .env."

    # Warn at startup / first call if a dev/trial key is detected.
    if TAVILY_API_KEY.startswith("tvly-dev-"):
        print(
            "  ⚠️  Tavily dev/trial key detected. "
            "Rate limits may apply; replace with a production key for reliable searches."
        )

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

            # Surface auth errors distinctly so the user knows it's a key problem.
            if response.status_code in (401, 403):
                print(f"⚠️  Tavily auth error ({response.status_code}): invalid or expired API key.")
                return (
                    "Web search unavailable: the Tavily API key is invalid or has expired. "
                    "Please update TAVILY_API_KEY in your .env file."
                )

            if response.status_code == 429:
                print("⚠️  Tavily rate limit reached.")
                return "Web search unavailable: rate limit reached. Please try again shortly."

            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "")
            if answer:
                return answer
            # Fall back to concatenating result snippets when no direct answer
            results = data.get("results", [])
            if results:
                snippets = [r.get("content", "") for r in results[:3]]
                return " ".join(snippets)
            return "No relevant information found via web search."
    except httpx.TimeoutException:
        return "Web search timed out. Please try again later."
    except Exception as e:
        print(f"⚠️  Web search error: {e}")
        return "Web search failed. The assistant will try to answer from its training knowledge instead."
