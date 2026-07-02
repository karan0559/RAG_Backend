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
        # On cloud hosts (e.g. HF Spaces) this frequently fails with an
        # SSLEOFError — YouTube blocking the datacenter IP at the TLS layer
        # rather than returning a clean HTTP error. See README's "Known
        # limitations" section.
        return [f" YouTube transcript failed: {e}"]

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")  
    elif "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [""])[0] 
    return ""
