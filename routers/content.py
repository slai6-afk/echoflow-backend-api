import os
import io
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.news_fetcher import fetch_tech_articles

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/articles")
async def get_articles(count: int = 3):
    articles = await fetch_tech_articles(count)
    return articles


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"  # alloy | echo | fable | onyx | nova | shimmer


@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Generate reference audio using OpenAI TTS (tts-1-hd).
    Falls back to a 400 error if OPENAI_API_KEY is not set,
    so the frontend can fall back to Web Speech API.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(503, "TTS not configured — set OPENAI_API_KEY")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=req.voice,
            input=req.text[:500],  # cap length
            response_format="mp3",
        )

        audio_bytes = response.read()
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except Exception as e:
        logger.error("TTS error: %s", e)
        raise HTTPException(500, f"TTS failed: {e}")


SEGMENT_WINDOW = 300  # 5 minutes per session

CURATED_VIDEOS = [
    {
        "id": "YUbSpI0J9aQ",
        "title": "UX Design Talk · Speaker 1",
        "description": "Clear professional delivery — ideal for design and product teams",
        "start_at": 30,
        "domain": "Design",
    },
    {
        "id": "cHuqhQmc4ok",
        "title": "UX Design Talk · Speaker 2",
        "description": "Conversational English with natural pacing and connected speech",
        "start_at": 30,
        "domain": "Design",
    },
    {
        "id": "eh8bcBIAAFo",
        "title": "UX Design Talk · Speaker 3",
        "description": "Precise vocabulary at a measured pace — strong sentence stress",
        "start_at": 30,
        "domain": "Design",
    },
]


@router.get("/youtube-clips")
async def get_youtube_clips():
    """Return curated video list with fetched transcript segments."""
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

    ytt = YouTubeTranscriptApi()
    clips = []

    for video in CURATED_VIDEOS:
        try:
            tl = ytt.list(video["id"])
            t = tl.find_transcript(["en"])
            raw = t.fetch()

            start_at = video.get("start_at", 0)
            end_at = start_at + SEGMENT_WINDOW

            # Merge short caption snippets into natural sentence-length chunks
            segments = []
            buf_text, buf_start, buf_end = "", 0.0, 0.0

            for snippet in raw:
                # Only process snippets within the 5-minute window
                if snippet.start < start_at:
                    continue
                if snippet.start >= end_at:
                    break
                text = snippet.text.strip().replace("\n", " ")
                start = snippet.start
                end = snippet.start + snippet.duration

                if not buf_text:
                    buf_text = text
                    buf_start = start
                    buf_end = end
                else:
                    buf_text += " " + text
                    buf_end = end

                # Flush on sentence boundary or when buffer is long enough
                is_sentence_end = any(buf_text.rstrip().endswith(c) for c in (".", "?", "!"))
                is_long = len(buf_text.split()) >= 12
                if is_sentence_end or is_long:
                    # Add a grace period so the player doesn't cut off the last word's audio
                    grace = 0.4 if is_sentence_end else 0.2
                    segments.append({
                        "text": buf_text.strip(),
                        "start": round(buf_start, 2),
                        "end": round(buf_end + grace, 2),
                    })
                    buf_text, buf_start, buf_end = "", 0.0, 0.0

            if buf_text:
                segments.append({"text": buf_text.strip(), "start": round(buf_start, 2), "end": round(buf_end + 0.3, 2)})

            clips.append({**video, "segments": segments[:25]})

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning("No transcript for %s: %s", video["id"], e)
            clips.append({**video, "segments": [], "error": "No transcript"})
        except Exception as e:
            logger.error("Transcript fetch error for %s: %s", video["id"], e)
            clips.append({**video, "segments": [], "error": str(e)})

    return clips
