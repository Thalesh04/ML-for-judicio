# api.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from ml_classifier import LaMiniAnalyzer   # <-- your class

app = FastAPI(
    title="LaMini Document Analyzer",
    description="Semantic search + key-clause extraction (EN/HI)",
    version="1.0.0"
)

# Load model once at startup (shared across requests)
analyzer = LaMiniAnalyzer(device="cpu")   # change to "cuda" if GPU available


class SearchRequest(BaseModel):
    text: str
    query: str
    top_k: int = 5
    max_chunk_sentences: int = 6
    overlap_sentences: int = 1


class ClauseRequest(BaseModel):
    text: str
    keywords: List[str] = None
    top_k: int = 5


@app.post("/search", response_model=List[Tuple[float, str]])
def semantic_search(req: SearchRequest):
    try:
        analyzer.fit_document(
            req.text,
            max_chunk_sentences=req.max_chunk_sentences,
            overlap_sentences=req.overlap_sentences,
        )
        return analyzer.semantic_search(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/key-clauses", response_model=List[Tuple[float, str]])
def key_clauses(req: ClauseRequest):
    try:
        return analyzer.extract_key_clauses(req.text, keywords=req.keywords, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)