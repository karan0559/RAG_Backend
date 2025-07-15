import requests
import trafilatura
from typing import List

def extract_from_url(url: str) -> List[str]:
    try:
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

        if response.status_code != 200:
            return [f" Failed to fetch URL. Status code: {response.status_code}"]

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False
        )

        return [text.strip()] if text else [" No meaningful text extracted."]

    except requests.exceptions.Timeout:
        return ["URL extraction timed out."]
    except requests.exceptions.RequestException as e:
        return [f" Request failed: {str(e)}"]
    except Exception as e:
        return [f" URL extraction failed: {str(e)}"]
