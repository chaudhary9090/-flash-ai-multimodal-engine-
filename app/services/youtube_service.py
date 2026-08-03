"""
YouTube Service: oEmbed Metadata & Knowledge Extraction.
Single-purpose design ready for LangChain Tool conversion in Phase 4.
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Optional
from app.core.logging import logger


def extract_youtube_id(text: str) -> Optional[str]:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def fetch_youtube_knowledge(url_or_prompt: str) -> Optional[str]:
    """
    Extracts YouTube video details and overview notes.
    
    Signature: (query: str) -> Optional[str]
    Single-purpose contract ready for Phase 4 @tool wrapping.
    """
    video_id = extract_youtube_id(url_or_prompt)
    if not video_id:
        return None

    full_yt_url = f"https://www.youtube.com/watch?v={video_id}"
    oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(full_yt_url)}&format=json"

    try:
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            title = data.get("title", "YouTube Video")
            author = data.get("author_name", "Unknown Creator")
            clean_title = re.sub(r"[^\w\s]", "", title.split("|")[0].split("-")[0]).strip()

            return (
                f"[YOUTUBE VIDEO KNOWLEDGE SUMMARY]\n"
                f"Title: {title}\n"
                f"Channel / Creator: {author}\n"
                f"Video Link: {full_yt_url}\n"
                f"--------------------------------------------------\n"
                f"SUMMARY & STUDY NOTES:\n"
                f"This video by '{author}' covers '{clean_title}'.\n"
                f"--------------------------------------------------"
            )

    except Exception as e:
        logger.error(f"YouTube tool error: {e}")
        return f"[YOUTUBE SUMMARY]\nVideo Link: {full_yt_url}\nStatus: Video link detected."
