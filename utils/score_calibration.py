"""
Score calibration middleware.

Raw Azure Pronunciation Assessment scores are calibrated against native-speaker baselines.
Even proficient native speakers score ~75–85 on Azure's strict phoneme model,
so raw scores systematically understate actual proficiency.

Calibration bands:
  Raw 75–100  →  Scaled 90–100  (Native Mastery)
  Raw 60–74   →  Scaled 75–89   (Fluent)
  Raw 40–59   →  Scaled 50–74   (Developing)
  Raw 0–39    →  Scaled 0–49    (Foundational)
"""


def calibrate_score(raw: float) -> float:
    """Map a raw Azure score to a human-meaningful calibrated score."""
    raw = max(0.0, min(100.0, raw))

    if raw >= 75:
        # 75–100 → 90–100
        return 90.0 + (raw - 75.0) / 25.0 * 10.0
    elif raw >= 60:
        # 60–74 → 75–89
        return 75.0 + (raw - 60.0) / 15.0 * 14.0
    elif raw >= 40:
        # 40–59 → 50–74
        return 50.0 + (raw - 40.0) / 20.0 * 24.0
    else:
        # 0–39 → 0–49
        return raw / 39.0 * 49.0


def calibrate_assessment(assessment: dict) -> dict:
    """
    Apply calibration to all score fields in an Azure assessment result.
    Returns a new dict with both raw_* and calibrated_* scores.
    """
    score_keys = [
        "accuracy_score",
        "fluency_score",
        "completeness_score",
        "pronunciation_score",
    ]

    result = dict(assessment)

    for key in score_keys:
        if key in result and result[key] is not None:
            raw = float(result[key])
            result[f"raw_{key}"] = raw
            result[key] = round(calibrate_score(raw), 1)

    # Calibrate word-level and phoneme-level scores
    if "words" in result:
        calibrated_words = []
        for word in result["words"]:
            w = dict(word)
            if "accuracy_score" in w and w["accuracy_score"] is not None:
                raw_w = float(w["accuracy_score"])
                w["raw_accuracy_score"] = raw_w
                w["accuracy_score"] = round(calibrate_score(raw_w), 1)

            if "phonemes" in w:
                calibrated_phonemes = []
                for ph in w["phonemes"]:
                    p = dict(ph)
                    if "accuracy_score" in p and p["accuracy_score"] is not None:
                        raw_p = float(p["accuracy_score"])
                        p["raw_accuracy_score"] = raw_p
                        p["accuracy_score"] = round(calibrate_score(raw_p), 1)
                    calibrated_phonemes.append(p)
                w["phonemes"] = calibrated_phonemes

            calibrated_words.append(w)
        result["words"] = calibrated_words

    return result
