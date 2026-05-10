import os
import json
import re
import logging
from typing import Optional
import anthropic

logger = logging.getLogger(__name__)


class LinguisticAnalyzer:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

    def _extract_worst_phonemes(self, assessment_data: dict) -> list[dict]:
        """Derive worst-scoring phonemes directly from Azure word/phoneme data."""
        phoneme_scores: dict[str, list[float]] = {}

        for word in assessment_data.get("words", []):
            for ph in word.get("phonemes", []):
                sym = ph.get("phoneme", "").strip()
                score = ph.get("accuracy_score")
                if sym and score is not None:
                    phoneme_scores.setdefault(sym, []).append(float(score))

        if not phoneme_scores:
            return []

        averaged = [
            {"phoneme": f"/{sym}/", "avg_score": round(sum(v) / len(v), 1), "count": len(v)}
            for sym, v in phoneme_scores.items()
        ]
        return sorted(averaged, key=lambda x: x["avg_score"])[:5]

    def generate_linguist_report(self, assessment_data: dict, native_language: str) -> dict:
        worst_phonemes = self._extract_worst_phonemes(assessment_data)

        if not self.client:
            return self._demo_report(native_language, worst_phonemes)

        worst_str = "\n".join(
            f"  - /{p['phoneme'].strip('/')}/ — avg score {p['avg_score']}/100 ({p['count']} occurrences)"
            for p in worst_phonemes
        ) or "  (no phoneme-level data available)"

        prompt = f"""You are analyzing real pronunciation assessment data for a tech professional.
Their native language is: {native_language}

ACTUAL worst-scoring IPA phonemes from Azure Speech (derived from real recording — use these, do NOT invent others):
{worst_str}

Full assessment JSON:
{json.dumps({k: v for k, v in assessment_data.items() if k != "words"}, indent=2)}

Generate a "Piercing Insight" report. The bottom_3_phonemes MUST use the exact IPA symbols listed above.
Be anatomical — explain the physical mechanics of each mistake in terms of {native_language} L1 interference.

Return ONLY a JSON object with these exact keys:
{{
  "signature_insight": "One piercing, personal observation referencing specific sounds from their actual data. Make it feel like a revelation.",
  "bottom_3_phonemes": [
    {{"phoneme": "/exact IPA from list above/", "note": "Anatomical reason tied to {native_language} L1 interference — what the tongue/jaw/lips are doing wrong"}},
    {{"phoneme": "/exact IPA from list above/", "note": "..."}},
    {{"phoneme": "/exact IPA from list above/", "note": "..."}}
  ],
  "first_exercise": "One specific drill targeting their actual worst phoneme — concrete and actionable",
  "overall_pattern": "2-sentence summary of the dominant pattern",
  "fluency_observations": "1 sentence on their rhythm and prosody"
}}

Tone: A world-class linguist who is precise and deeply encouraging. No generic advice."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system="You are a PhD Linguist specializing in L1-to-L2 phonological interference for tech professionals. You provide precise, anatomically-grounded, empathetic insights. Always respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"raw": raw}
        except Exception as e:
            logger.error("Linguist report error: %s", e)
            return self._demo_report(native_language)

    def analyze_import_full(self, transcript: str, assessment: dict, role: str, native_language: str) -> dict:
        """4-dimension analysis: transcript, content, tone, pronunciation."""
        phoneme_concentration = self._extract_phoneme_concentration(assessment)
        llm = self._analyze_content_and_tone(transcript, role, native_language)
        return {
            "transcript": transcript,
            "content_analysis": llm.get("content_analysis", {}),
            "tone_analysis": llm.get("tone_analysis", {}),
            "pronunciation": {
                "overall_score": assessment.get("pronunciation_score"),
                "fluency_score": assessment.get("fluency_score"),
                "completeness_score": assessment.get("completeness_score"),
                "accuracy_score": assessment.get("accuracy_score"),
                "word_scores": assessment.get("words", []),
                "phoneme_concentration": phoneme_concentration,
            },
        }

    def _extract_phoneme_concentration(self, assessment: dict) -> list[dict]:
        phoneme_scores: dict[str, list[float]] = {}
        phoneme_words: dict[str, list[str]] = {}
        for word in assessment.get("words", []):
            word_text = word.get("word", "")
            for ph in word.get("phonemes", []):
                sym = ph.get("phoneme", "").strip()
                score = ph.get("accuracy_score")
                if sym and score is not None:
                    phoneme_scores.setdefault(sym, []).append(float(score))
                    if word_text and word_text not in phoneme_words.get(sym, []):
                        phoneme_words.setdefault(sym, []).append(word_text)
        if not phoneme_scores:
            return []
        result = [
            {
                "phoneme": f"/{sym}/",
                "avg_score": round(sum(scores) / len(scores), 1),
                "count": len(scores),
                "example_words": phoneme_words.get(sym, [])[:3],
            }
            for sym, scores in phoneme_scores.items()
        ]
        return sorted(result, key=lambda x: x["avg_score"])[:8]

    def _analyze_content_and_tone(self, transcript: str, role: str, native_language: str) -> dict:
        if not transcript:
            return self._demo_content_tone()
        if not self.client:
            return self._demo_content_tone()

        prompt = f"""Analyze this professional's speech as a native North American English speaker and senior {role} at a major tech company.

SPEAKER: Role = {role}, Native language = {native_language or "unknown"}

TRANSCRIPT:
{transcript}

Return TWO analyses as valid JSON:

{{
  "content_analysis": {{
    "unnatural_phrases": [
      {{"original": "exact quote from transcript", "issue": "why it sounds non-native or unidiomatic", "alternative": "how a native {role} would say it"}}
    ],
    "word_choice_issues": [
      {{"original": "exact quote", "issue": "too formal/casual/wrong for context", "alternative": "better option"}}
    ],
    "filler_words": ["list e.g. 'um (4x)', 'basically (3x)', 'like (6x)'"],
    "domain_opportunities": ["e.g. 'Said X — a {role} would say Y'"],
    "overall_content_score": 0-100
  }},
  "tone_analysis": {{
    "tone_label": "single phrase e.g. 'Flat and hesitant' or 'Confident but rushed'",
    "energy_level": "Low | Medium | High",
    "storytelling_quality": "Poor | Developing | Good | Excellent",
    "hedging_patterns": ["exact phrases like 'I think maybe', 'kind of sort of'"],
    "verbal_habits": ["recurring issues e.g. 'Starts 7 sentences with So', 'Trails off mid-sentence'"],
    "strengths": ["2-3 specific things they do well"],
    "key_recommendation": "One actionable change with a concrete technique"
  }}
}}

Be specific — quote the actual transcript. As a native speaker, flag things that would make you wince even if grammatically correct."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=f"You are a native North American English speaker, expert communication coach, and senior {role} who has trained hundreds of non-native professionals. Be precise, direct, and kind. Return valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._demo_content_tone()
        except Exception as e:
            logger.error("Content/tone analysis error: %s", e)
            return self._demo_content_tone()

    def _demo_content_tone(self) -> dict:
        return {
            "content_analysis": {
                "unnatural_phrases": [
                    {"original": "I want to discuss about the design", "issue": "'discuss about' is non-idiomatic — 'about' is redundant after 'discuss'", "alternative": "I want to discuss the design"},
                    {"original": "From my side, I think we should", "issue": "'From my side' is a literal translation pattern — sounds non-native", "alternative": "My take is that we should / In my view"},
                ],
                "word_choice_issues": [
                    {"original": "very unique", "issue": "'Unique' is absolute — 'very unique' is grammatically incorrect and sounds informal", "alternative": "distinctive / quite unusual"},
                ],
                "filler_words": ["um (3x)", "basically (4x)", "you know (2x)"],
                "domain_opportunities": ["Said 'make it look better' — a senior UX designer would say 'improve visual hierarchy' or 'reduce cognitive load'"],
                "overall_content_score": 68,
            },
            "tone_analysis": {
                "tone_label": "Hesitant but knowledgeable",
                "energy_level": "Medium",
                "storytelling_quality": "Developing",
                "hedging_patterns": ["'I think maybe we could'", "'I'm not sure but possibly'", "'kind of like'"],
                "verbal_habits": ["Starts most sentences with 'So'", "Trails off before reaching the main point"],
                "strengths": ["Clear logical structure", "Good vocabulary range", "Appropriate professional register"],
                "key_recommendation": "Replace hedging openers with direct statements. Instead of 'I think maybe we should consider...', say 'My recommendation is...'. Record yourself and count how often you start with 'So' — aim for zero.",
            },
        }

    def generate_import_analysis(self, transcript: str, native_language: str) -> dict:
        if not self.client:
            return self._demo_import_analysis()

        prompt = f"""Analyze this speech transcript from a tech professional (native language: {native_language}).

Transcript:
{transcript}

Identify the "high-stakes moments" — places where pronunciation, pacing, or prosody undermined professional authority.

Return ONLY JSON:
{{
  "overall_score": 0-100,
  "summary": "2-3 sentence overall assessment. Be direct and empathetic.",
  "high_stakes_moments": [
    {{
      "timestamp": "estimated position like '0:30-0:45' or 'opening'",
      "insight": "What happened and why it undermines authority",
      "recommendation": "Specific corrective technique"
    }}
  ]
}}"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system="You are a PhD Linguist and executive communication coach. Analyze speech for pronunciation and prosody issues. Always respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"raw": raw}
        except Exception as e:
            logger.error("Import analysis error: %s", e)
            return self._demo_import_analysis()

    def chat_reply(self, message: str, history: list, native_language: str = "") -> str:
        if not self.client:
            return self._demo_chat_reply(len(history))

        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        msgs.append({"role": "user", "content": message})

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=f"""You are an empathetic AI linguist conducting a diagnostic conversation with a tech professional.
Your goal: understand their communication context to identify L1 interference patterns.
Ask about their role, their most stressful communication situations, and when they feel least confident.
Keep responses under 3 sentences. Be warm, curious, and insightful.
{f'Their native language is {native_language}.' if native_language else ''}
After 3 user messages, signal readiness to generate their report by saying "I have everything I need."
""",
                messages=msgs,
            )
            return response.content[0].text
        except Exception as e:
            logger.error("Chat error: %s", e)
            return self._demo_chat_reply(len(history))

    def generate_phoneme_insights(self, error_profile: dict, native_language: str) -> list[dict]:
        """Generate FocusAreaCard[] from a SortedErrorProfile via LLM."""
        top_phonemes = error_profile.get("phonemes", [])[:3]

        if not top_phonemes:
            return self._demo_focus_cards(native_language)

        if not self.client:
            return self._demo_focus_cards(native_language)

        # Build a compact, data-dense phoneme block for the prompt
        lines = []
        for p in top_phonemes:
            rate = p.get("mispronunciationRate", 0)
            line = (
                f"  - {p['symbol']} — avg {p['avgScore']}/100 "
                f"across {p['count']} occurrences"
            )
            if rate > 0.25:
                line += f", {round(rate * 100)}% mispronunciation rate"
            if p.get("likelySub"):
                line += f", substituted as {p['likelySub']}"
            if p.get("exampleWords"):
                line += f" (heard in: {', '.join(p['exampleWords'])})"
            lines.append(line)

        phoneme_block = "\n".join(lines)

        prompt = f"""You are a PhD linguist specializing in L2 phonological acquisition for tech professionals.

SPEAKER DATA:
- Native language: {native_language}
- Overall pronunciation accuracy: {error_profile.get('overallAccuracy', '?')}/100
- Phoneme occurrences analyzed: {error_profile.get('totalPhonemeOccurrences', '?')}

TOP 3 PROBLEM PHONEMES (worst first):
{phoneme_block}

Generate exactly 3 FocusAreaCard objects as a JSON array. For EACH phoneme:

  "phonemeSymbol": the exact IPA symbol shown above (e.g. "/θ/")
  "phoneticCategory": technical label (e.g. "Dental Fricative", "Low Front Vowel", "Retroflex Approximant")
  "linguisticCause": 2 sentences MAXIMUM.
      Sentence 1: The anatomical failure — what specifically the tongue tip/body/dorsum, jaw, lips, or velum are doing wrong.
      Sentence 2: Why {native_language} L1 transfer causes this exact error — reference the L1 phonological inventory gap or allophonic pattern, not a generic "L1 interference" statement.
  "avgScore": copy from input data
  "substitution": ONLY include if the data shows a clear substitution pattern (e.g. "/ɹ/ → /l/"). Omit the key entirely otherwise.

CONSTRAINTS:
- Never write "needs practice" or "common error" — be anatomically precise.
- The L1 explanation must name a specific phonological feature of {native_language}: a missing phoneme category, a different coda strategy, an allophonic merger, etc.
- If substitution data is present, the linguisticCause MUST address the tongue-body positioning difference between the target and the substituted phoneme.

Return ONLY a valid JSON array of 3 objects. No markdown fences, no commentary."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system="You are a PhD Linguist specializing in L1-to-L2 phonological interference. Return valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return self._demo_focus_cards(native_language)
        except Exception as e:
            logger.error("Phoneme insights error: %s", e)
            return self._demo_focus_cards(native_language)

    def _demo_focus_cards(self, native_language: str) -> list[dict]:
        cards: dict[str, list[dict]] = {
            "Mandarin Chinese": [
                {
                    "phonemeSymbol": "/θ/",
                    "phoneticCategory": "Dental Fricative",
                    "linguisticCause": "The tongue tip must protrude between the upper and lower teeth, creating lamino-dental friction against the cutting edges — a placement that Mandarin speakers consistently avoid by retracting the tip to the alveolar ridge. Mandarin has no dental fricatives whatsoever; its inventory jumps from bilabials directly to alveolar /s/ and /ts/, so the inter-dental gesture has zero L1 scaffolding.",
                    "avgScore": 42.0,
                },
                {
                    "phonemeSymbol": "/ɹ/",
                    "phoneticCategory": "Retroflex Approximant",
                    "linguisticCause": "English /ɹ/ requires the tongue body to bunch upward and backward with no palatal contact, accompanied by slight lip rounding — an approximant with no friction. Mandarin /r/ (IPA /ʐ/) is a retroflex fricative where the tongue tip curls to contact the post-alveolar ridge, producing audible friction, so Mandarin speakers over-curl and over-constrict, producing a buzzy quality English listeners perceive as mispronounced.",
                    "avgScore": 51.3,
                },
                {
                    "phonemeSymbol": "/æ/",
                    "phoneticCategory": "Low Front Tense Vowel",
                    "linguisticCause": "The trap vowel demands maximum jaw aperture combined with a front-advanced tongue body and spread, unrounded lips — a sustained, effortful posture. Mandarin's /a/ is a low central vowel; the tongue body sits in a neutral central position rather than advancing toward the lower front teeth, so the perceived vowel lands as /ɛ/ or /ɑ/ rather than /æ/.",
                    "avgScore": 58.7,
                },
            ],
            "Spanish": [
                {
                    "phonemeSymbol": "/v/",
                    "phoneticCategory": "Labiodental Fricative",
                    "linguisticCause": "English /v/ requires the inner lower lip to rest against the upper incisors while voicing generates labiodental friction continuously through the segment. Spanish merges /b/ and /v/ as a single phoneme with bilabial realization — in intervocalic position it surfaces as the bilabial approximant /β/ — so the upper-teeth-to-lower-lip contact is entirely absent from the L1 motor program.",
                    "avgScore": 44.5,
                },
                {
                    "phonemeSymbol": "/θ/",
                    "phoneticCategory": "Dental Fricative",
                    "linguisticCause": "The tongue tip must sustain contact with the upper incisors' cutting edges while voiceless air streams around it. Latin American Spanish completely lacks /θ/ through seseo merger with /s/, so speakers default to alveolar /s/ rather than the more anterior dental gesture; even Castilian-influenced speakers use a more laminal, less protruded contact than the English apical target.",
                    "avgScore": 55.1,
                },
                {
                    "phonemeSymbol": "/ŋ/",
                    "phoneticCategory": "Velar Nasal",
                    "linguisticCause": "Word-final /ŋ/ requires the tongue dorsum to seal against the velum while nasal airflow continues without any following stop release. Spanish /n/ assimilates to velar position only allophonically before /k/ and /ɡ/ and never appears word-finally as a standalone phoneme, so speakers unconsciously append a released velar stop /ɡ/ or substitute alveolar /n/, both of which collapse the coda nasality.",
                    "avgScore": 61.2,
                },
            ],
        }

        default = [
            {
                "phonemeSymbol": "/θ/",
                "phoneticCategory": "Dental Fricative",
                "linguisticCause": "The tongue tip must contact the cutting edge of the upper incisors while voiceless air escapes over the tongue's dorsal surface — inter-dental placement absent from most world phonological inventories. Without this gesture in the L1 motor repertoire, speakers default to the nearest alveolar fricative /s/ or stop /t/, both of which fall short of the dental target.",
                "avgScore": 48.0,
            },
            {
                "phonemeSymbol": "/æ/",
                "phoneticCategory": "Low Front Tense Vowel",
                "linguisticCause": "The jaw must open to maximum aperture while the tongue body advances toward the lower front teeth, with unrounded lips held spread — a posture requiring sustained muscular effort. Most L1 systems use a central low /a/ for their open vowel, leaving the forward tongue displacement unautomatized and producing a centralized /ɛ/ or /ɑ/ instead.",
                "avgScore": 57.5,
            },
            {
                "phonemeSymbol": "/ɹ/",
                "phoneticCategory": "Retroflex Approximant",
                "linguisticCause": "The tongue body must bunch upward and posteriorly with absolutely no surface contact against the palate, while the lips round slightly — a pure approximant with no friction whatsoever. Speakers whose L1 /r/ involves a trill, tap, or fricative (all requiring tongue-palate contact) over-constrict the passage, introducing unwanted friction that English listeners perceive as non-native.",
                "avgScore": 63.2,
            },
        ]

        return cards.get(native_language, default)

    def _demo_report(self, native_language: str, worst_phonemes: Optional[list] = None) -> dict:
        lang_map = {
            "Mandarin Chinese": {
                "phonemes": ["/θ/", "/ɹ/", "/æ/"],
                "notes": [
                    "Mandarin lacks dental fricatives — your tongue tip stays too far back. It needs to touch the upper teeth edge.",
                    "Mandarin /r/ is a retroflex; English /ɹ/ requires tongue bunching, not touching the palate.",
                    "The /æ/ trap vowel doesn't exist in Mandarin — your jaw isn't dropping far enough.",
                ],
            },
            "Spanish": {
                "phonemes": ["/v/", "/θ/", "/h/"],
                "notes": [
                    "Spanish /b/ and /v/ are allophones — your English /v/ is missing the labiodental friction.",
                    "The dental /θ/ in 'think' sounds like /s/ or /t/ because Spanish uses different coda strategies.",
                    "English /h/ requires glottal constriction that Spanish-speakers often omit or over-aspirate.",
                ],
            },
        }

        lang_data = lang_map.get(
            native_language,
            {
                "phonemes": ["/θ/", "/æ/", "/ɹ/"],
                "notes": [
                    "Dental fricative — tongue position is slightly too far back from the teeth.",
                    "Trap vowel — jaw aperture is not opening wide enough for native-level clarity.",
                    "English approximant — slight L1 interference in the tongue body position.",
                ],
            },
        )

        # Use real phoneme data if available, fall back to L1-based defaults
        if worst_phonemes:
            bottom_3 = [
                {"phoneme": p["phoneme"], "note": f"Scored {p['avg_score']}/100 — likely {native_language} L1 interference affecting articulation"}
                for p in worst_phonemes[:3]
            ]
        else:
            bottom_3 = [
                {"phoneme": lang_data["phonemes"][i], "note": lang_data["notes"][i]}
                for i in range(3)
            ]

        return {
            "signature_insight": f"Your rhythm is confident, but you're substituting your L1 coda patterns when words end in consonant clusters. In '{native_language}', this simplification is natural — in English, it signals hesitation to listeners.",
            "bottom_3_phonemes": bottom_3,
            "first_exercise": "Start with 5 minutes of minimal pair drills on your first phoneme — record yourself, then listen back at 0.8x speed. The slowed playback makes the gap between your production and the target unmistakable.",
            "overall_pattern": "Strong lexical command and fluency, but phoneme-level precision drops under cognitive load. This is the pattern of someone who learned English academically — the foundation is excellent, the fine-tuning is what we'll address.",
            "fluency_observations": "Your natural speaking tempo is well-calibrated; the issue is stress placement on polysyllabic technical terms.",
        }

    def _demo_import_analysis(self) -> dict:
        return {
            "overall_score": 71,
            "summary": "Your delivery was authoritative in the opening, but your pace accelerated noticeably during complex technical explanations — a sign your L1 prosodic rhythm took over under cognitive load. The conclusion felt rushed, which undermines the perceived confidence of your closing argument.",
            "high_stakes_moments": [
                {
                    "timestamp": "Early section",
                    "insight": "You rushed through multi-syllabic technical terms, compressing vowels in a way that signals anxiety to native listeners.",
                    "recommendation": "Slow to 85% of your natural rate when introducing technical concepts. Stress the primary syllable more deliberately.",
                },
                {
                    "timestamp": "Middle section",
                    "insight": "Sentence-final intonation dropped sharply, creating a perception of uncertainty despite confident word choice.",
                    "recommendation": "Use a sustained, gradual fall rather than an abrupt drop. Practice with a mirror watching your jaw position.",
                },
                {
                    "timestamp": "Conclusion",
                    "insight": "Your vowels compressed significantly, stripping away the tonal authority of your closing statement.",
                    "recommendation": "Expand vowel duration by 20% in your final sentence. Breathe before your conclusion — the pause signals confidence.",
                },
            ],
        }

    def _demo_chat_reply(self, history_len: int) -> str:
        responses = [
            "That's a high-pressure communication environment. When you present to stakeholders, do you feel more comfortable with prepared remarks or spontaneous Q&A?",
            "Very telling — most L2 speakers find Q&A harder because it removes the prosodic scaffolding of prepared speech. What type of technical content do you present most often?",
            "I have everything I need. I can already see some interesting patterns in the way you describe your challenges — your self-awareness is actually a significant advantage in this process.",
        ]
        idx = min(history_len // 2, len(responses) - 1)
        return responses[idx]
