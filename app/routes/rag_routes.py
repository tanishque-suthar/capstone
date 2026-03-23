"""
FastAPI router for Track 3 RAG endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.pipeline.rag import get_rag_pipeline

router = APIRouter(prefix="/api/rag", tags=["RAG (Track 3)"])

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@router.post("/ingest/{event_id}")
async def ingest_event(event_id: str):
    """Ingest all image crops for a specific event into LanceDB."""
    try:
        pipeline = get_rag_pipeline()
        count = pipeline.ingest_event_crops(event_id)
        return {"status": "success", "event_id": event_id, "ingested_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search(request: SearchRequest):
    """Perform a semantic search against the LanceDB entity crops using natural language."""
    try:
        pipeline = get_rag_pipeline()
        results = pipeline.search_crops(request.query, limit=request.limit)
        return {"status": "success", "query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
