import os
import tempfile
import logging

logger = logging.getLogger(__name__)


_DEMO_REF = (
    "So basically I want to discuss the new design system we are working on. "
    "From my side I think maybe we should consider the user flow more carefully. "
    "The interface should be very unique and make it look better for our users. "
    "I am not sure but possibly we could improve it."
)


class AzureSpeechService:
    def __init__(self):
        self.key = os.getenv("AZURE_SPEECH_KEY", "")
        self.region = os.getenv("AZURE_SPEECH_REGION", "eastus")

    def is_configured(self) -> bool:
        return bool(self.key)

    async def evaluate_pronunciation(self, audio_data: bytes, reference_text: str) -> dict:
        if not self.is_configured():
            return self._demo_result(reference_text)

        try:
            import azure.cognitiveservices.speech as speechsdk

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            speech_config = speechsdk.SpeechConfig(
                subscription=self.key, region=self.region
            )
            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                enable_miscue=True,
            )
            pronunciation_config.phoneme_alphabet = "IPA"

            audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )
            pronunciation_config.apply_to(recognizer)

            result = recognizer.recognize_once()

            os.unlink(tmp_path)

            if result.reason.name == "RecognizedSpeech":
                assessment = speechsdk.PronunciationAssessmentResult(result)
                words = []
                for word in assessment.words:
                    phonemes = [
                        {
                            "phoneme": p.phoneme,
                            "accuracy_score": p.accuracy_score,
                            "offset": p.offset,
                            "duration": p.duration,
                        }
                        for p in word.phonemes
                    ]
                    words.append(
                        {
                            "word": word.word,
                            "accuracy_score": word.accuracy_score,
                            "error_type": word.error_type,
                            "phonemes": phonemes,
                        }
                    )

                return {
                    "accuracy_score": assessment.accuracy_score,
                    "fluency_score": assessment.fluency_score,
                    "completeness_score": assessment.completeness_score,
                    "pronunciation_score": assessment.pronunciation_score,
                    "words": words,
                    "recognized_text": result.text,
                }
            else:
                logger.warning("Azure Speech recognition failed: %s", result.reason)
                return self._demo_result(reference_text)

        except Exception as e:
            logger.error("Azure Speech error: %s", e)
            return self._demo_result(reference_text)

    def _demo_result(self, reference_text: str) -> dict:
        import random

        # Common English phoneme set for realistic demo data
        PHONEMES = ["p","b","t","d","k","ɡ","f","v","θ","ð","s","z","ʃ","ʒ","h","tʃ","dʒ","m","n","ŋ","l","r","j","w","æ","ɑ","ɛ","ɪ","ɒ","ʌ","ʊ","iː","uː","eɪ","aɪ","ɔɪ","aʊ","oʊ"]

        # Use demo transcript when no reference text is provided (import without Whisper)
        text = reference_text.strip() if reference_text.strip() else _DEMO_REF
        words = text.split()
        word_results = []
        for w in words:
            acc = random.uniform(55, 98)
            error_type = "None" if acc > 70 else "Mispronunciation"
            # Generate 2–4 demo phonemes per word proportional to word length
            n_phonemes = min(max(len(w) // 2, 2), 5)
            phonemes = []
            for _ in range(n_phonemes):
                ph_score = random.uniform(50, 100) if error_type == "Mispronunciation" else random.uniform(75, 100)
                phonemes.append({
                    "phoneme": random.choice(PHONEMES),
                    "accuracy_score": ph_score,
                })
            word_results.append(
                {
                    "word": w,
                    "accuracy_score": acc,
                    "error_type": error_type,
                    "phonemes": phonemes,
                }
            )

        return {
            "accuracy_score": random.uniform(62, 88),
            "fluency_score": random.uniform(70, 92),
            "completeness_score": random.uniform(85, 100),
            "pronunciation_score": random.uniform(65, 85),
            "words": word_results,
            "recognized_text": text,
            "_demo": True,
        }
