import subprocess
import tempfile
import os
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from services.azure_speech import AzureSpeechService
from utils.score_calibration import calibrate_assessment

router = APIRouter()
speech_service = AzureSpeechService()
logger = logging.getLogger(__name__)


def convert_to_wav(audio_bytes: bytes, content_type: str, max_seconds: int = 0) -> bytes:
    """Convert any audio format to WAV using ffmpeg.

    max_seconds: if > 0, truncate to that duration before processing.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as inp:
            inp.write(audio_bytes)
            inp_path = inp.name

        out_path = inp_path.replace(".webm", ".wav")
        cmd = ["ffmpeg", "-y", "-i", inp_path]
        if max_seconds > 0:
            cmd.extend(["-t", str(max_seconds)])
        cmd.extend([
            "-ar", "16000", "-ac", "1",
            # highpass: remove low-frequency rumble (<80 Hz)
            # afftdn: FFT-based noise reduction
            # loudnorm: normalize perceived loudness for Azure
            "-af", "highpass=f=80,afftdn=nf=-20,loudnorm=I=-16:TP=-1.5:LRA=11",
            out_path,
        ])
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        os.unlink(inp_path)

        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                wav_bytes = f.read()
            os.unlink(out_path)
            return wav_bytes
        else:
            logger.warning("ffmpeg conversion failed, using raw bytes")
            return audio_bytes
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Audio conversion error: %s", e)
        return audio_bytes


@router.post("/evaluate")
async def evaluate_pronunciation(
    audio: UploadFile = File(...),
    reference_text: str = Form(...),
):
    if not reference_text.strip():
        raise HTTPException(400, "reference_text is required")

    audio_bytes = await audio.read()
    wav_bytes = convert_to_wav(audio_bytes, audio.content_type or "")
    result = await speech_service.evaluate_pronunciation(wav_bytes, reference_text)
    return calibrate_assessment(result)


async def _transcribe_with_whisper(wav_bytes: bytes) -> str:
    """Use OpenAI Whisper (async) for accurate transcription before pronunciation scoring."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                    # Faster response; language hint avoids detection overhead
                    language="en",
                )
            return result.strip() if isinstance(result, str) else result.text.strip()
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        return ""


@router.post("/analyze-import")
async def analyze_import(
    audio: UploadFile = File(...),
    role: str = Form("UX Designer"),
    native_language: str = Form(""),
):
    from services.linguistic_analyzer import LinguisticAnalyzer

    audio_bytes = await audio.read()
    # Trim to first 5 minutes — keeps latency predictable and Whisper/Azure reliable
    wav_bytes = convert_to_wav(audio_bytes, audio.content_type or "", max_seconds=300)

    # Step 1: Whisper transcription (most accurate — uses actual speech content)
    transcript = await _transcribe_with_whisper(wav_bytes)

    # Step 2: Azure Pronunciation Assessment
    # Always run regardless of whether Whisper succeeded.
    # - With transcript: uses it as reference for precise word/phoneme scoring.
    # - Without transcript: Azure demo mode uses a realistic fallback text so
    #   the UI always has content to display.
    assessment: dict = {}
    try:
        raw = await speech_service.evaluate_pronunciation(wav_bytes, transcript or "")
        assessment = calibrate_assessment(raw)
    except Exception as e:
        logger.warning("Pronunciation assessment failed: %s", e)

    # If Whisper had no result, use whatever Azure recognized
    if not transcript:
        transcript = assessment.get("recognized_text", "")

    # Step 3: Full 4-dimension analysis
    analyzer = LinguisticAnalyzer()
    return analyzer.analyze_import_full(
        transcript=transcript,
        assessment=assessment,
        role=role,
        native_language=native_language,
    )
