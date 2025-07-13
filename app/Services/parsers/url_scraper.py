import trafilatura


def extract_from_url(url: str):
    try:
        downloaded = trafilatura.fetch_url(url)  # ❌ remove timeout
        if not downloaded:
            return ["Failed to fetch content from URL."]

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text.split("\n") if text else ["No meaningful text extracted."]

    except Exception as e:
        return [f"URL extraction failed: {str(e)}"]
