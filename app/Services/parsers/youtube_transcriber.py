from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def extract_youtube_transcript(url: str):
    try:
        video_id = extract_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return [segment["text"] for segment in transcript if segment["text"].strip()]
    except Exception as e:
        return [f"YouTube transcript failed: {e}"]

def extract_video_id(url):
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path[1:]
    elif "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [""])[0]
    return ""
