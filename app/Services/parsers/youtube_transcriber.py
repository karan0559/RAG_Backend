from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def extract_youtube_transcript(url: str):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return [" Failed to extract video ID."]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [segment["text"] for segment in transcript if segment["text"].strip()]
    except Exception as e:
        return [f" YouTube transcript failed: {e}"]

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")  
    elif "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [""])[0] 
    return ""
