from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def extract_youtube_transcript(url: str):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return [" Failed to extract video ID."]
        # get_transcript() is a deprecated classmethod removed in some
        # library versions; fetch() is the current instance-based API.
        transcript = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
        return [segment["text"] for segment in transcript if segment["text"].strip()]
    except Exception as e:
        # TEMPORARY diagnostic print — remove once the cloud-host failure
        # mode is confirmed (nothing currently logs the raw exception before
        # the extractor's garbage-text filter discards it).
        print(f"[YouTubeTranscriber] RAW ERROR for {url}: {type(e).__name__}: {e}")
        return [f" YouTube transcript failed: {e}"]

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")  
    elif "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [""])[0] 
    return ""
