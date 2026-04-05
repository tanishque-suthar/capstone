"""
FastAPI router for causal reasoning and event Q&A endpoints.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.database import get_event
from app.models import ReasoningAnswer, ReasoningQuestionRequest, ReasoningReportResponse
from app.pipeline.reasoning import answer_question, load_reasoning_report

router = APIRouter(prefix="/api/reasoning", tags=["Reasoning"])


def _load_event_report(event_id: str):
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    path = event.get("Reasoning_JSON_Path")
    if not path:
        raise HTTPException(status_code=404, detail="Reasoning report not generated for this event")

    report_path = Path(path)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Reasoning report file is missing on disk")

    return load_reasoning_report(report_path)


@router.get("/{event_id}", response_model=ReasoningReportResponse)
async def get_reasoning_report(event_id: str):
    """Return the structured reasoning report for an event."""
    report = _load_event_report(event_id)
    return ReasoningReportResponse(
        event_id=report.event_id,
        trigger_time=report.trigger_time,
        summary=report.summary,
        objects=[item.__dict__ for item in report.objects],
        anomalies=[item.__dict__ for item in report.anomalies],
        hypotheses=[item.__dict__ for item in report.hypotheses],
        causal_graph=[item.__dict__ for item in report.causal_graph],
        multimodal_findings=[item.__dict__ for item in report.multimodal_findings],
        causal_engine=report.causal_engine.__dict__,
        confidence_gate=report.confidence_gate.__dict__,
    )


@router.post("/{event_id}/ask", response_model=ReasoningAnswer)
async def ask_reasoning_question(event_id: str, request: ReasoningQuestionRequest):
    """Answer a grounded question about an event."""
    report = _load_event_report(event_id)
    result = answer_question(report, request.question)
    return ReasoningAnswer(**result)
