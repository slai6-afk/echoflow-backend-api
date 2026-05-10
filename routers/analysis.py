from fastapi import APIRouter
from pydantic import BaseModel
from services.linguistic_analyzer import LinguisticAnalyzer

router = APIRouter()
analyzer = LinguisticAnalyzer()


class LinguistReportRequest(BaseModel):
    assessment_data: dict
    native_language: str


class PhonemeInsightsRequest(BaseModel):
    error_profile: dict
    native_language: str


class ChatRequest(BaseModel):
    message: str
    history: list[dict]
    native_language: str = ""


@router.post("/linguist-report")
def linguist_report(req: LinguistReportRequest):
    return analyzer.generate_linguist_report(req.assessment_data, req.native_language)


@router.post("/phoneme-insights")
def phoneme_insights(req: PhonemeInsightsRequest):
    return analyzer.generate_phoneme_insights(req.error_profile, req.native_language)


@router.post("/chat")
def chat(req: ChatRequest):
    reply = analyzer.chat_reply(req.message, req.history, req.native_language)
    return {"reply": reply}
