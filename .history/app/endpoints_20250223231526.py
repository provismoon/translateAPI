from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import json
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path
from app.processing import translate_processing, translate_processing_with_streaming
from app.instructions import get_instructions
from app.monitoring import MemoryTracker
from typing import AsyncGenerator
import asyncio


# Load report.json
#-----------이후 삭제
report_data = []
try:
    with open("report.json", "r") as f:
        report_data = json.load(f)
except Exception as e:
    print("Error loading report.json:", e)
#-----------이후 삭제

def get_router(model, tokenizer):
    router = APIRouter()

    class TranslationText(BaseModel):
        reportName: str = Field(description="제목")
        vulnerabilityDetail: str = Field(description="취약점")
        howToPatchDetail: str = Field(description="조치 사항")

    class TranslationRequest(TranslationText):
        origin_lang: str = Field(None, description="~로 부터 번역")
        target_lang: str = Field(description="~로 번역") # 타겟 lang에 국가 리스트 제한해서 매칭
    
        @validator("target_lang", pre=True)
        def validate_language(cls, value):
            allowed_languages = {"english", "hindi", "arabic", "russian", "japanese", "spanish", "french", "german"}
            value_lower = value.lower()
            if value_lower not in allowed_languages:
                raise ValueError(f"Language must be one of {', '.join(allowed_languages).title()} (case-insensitive)")
            return value_lower

    class TranslationResponse(TranslationText):
        origin_lang: str = Field(None, description="~로 부터 번역") # 감지된 언어. llm 한번 더 요청이라 생략
        target_lang: str = Field(description="~로 번역") # 타겟 lang에 국가 리스트 제한해서 매칭

    # 데모 시연 용 ---
    @router.get("/demo", response_class=HTMLResponse)
    def demo_page():
        return Path("static/demo.html").read_text()
    
    @router.get("/demo2", response_class=HTMLResponse)
    def demo_page():
        return Path("static/demo2.html").read_text()
        
    @router.get("/api/v1/report/{index}")
    def get_report(index: int):
        if index < 0 or index >= len(report_data):
            raise HTTPException(status_code=404, detail="Report not found")
        return report_data[index]

    @router.get("/api/v1/report")
    def get_all_reports():
        if not report_data:
            raise HTTPException(status_code=404, detail="No reports found")
        return report_data
    # 데모 시연 용 ---
    
    # 번역 라우터
    @router.post("/api/v1/translate", response_model=TranslationResponse)
    def translate_text(request: TranslationRequest):
        
        # 메모리 사용량 로깅
        tracker = MemoryTracker()
        tracker.log_memory("시작")
        
        tgt_lang = request.target_lang
        # Get instructions from the module
        instruction_report, instruction_general = get_instructions(tgt_lang)

        translated_report_name = translate_processing(model, tokenizer, instruction_report, request.reportName)
        translated_vulnerability_detail = translate_processing(model, tokenizer, instruction_general, request.vulnerabilityDetail)
        translated_how_to_patch_detail = translate_processing(model, tokenizer, instruction_general, request.howToPatchDetail)

        # 다시 로깅
        tracker.log_memory("추론 후")
        tracker.clear_memory()
        
        return TranslationResponse(
            reportName=translated_report_name,
            vulnerabilityDetail=translated_vulnerability_detail,
            howToPatchDetail=translated_how_to_patch_detail,
            target_lang=tgt_lang
        )
    
    
    return router
    