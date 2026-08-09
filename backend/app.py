"""
app.py
------
FastAPI backend for TOP. One real endpoint: POST /api/recommend, which
takes a live scorecard and returns the DecisionEngine's recommendation.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
Then open frontend/index.html (it calls http://localhost:8000/api/recommend).
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision_engine"))
from live.recommend import recommend  # noqa: E402

app = FastAPI(title="TOP - Tactical Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only - restrict this before deploying anywhere real
    allow_methods=["*"],
    allow_headers=["*"],
)


class Ball(BaseModel):
    over: int
    ball_in_over: int
    batter: str
    non_striker: str
    bowler: str
    batter_runs: int = 0
    wides: int = 0
    noballs: int = 0
    byes: int = 0
    legbyes: int = 0
    is_wicket: int = 0


class RecommendRequest(BaseModel):
    role: Literal["bowling", "batting"]
    venue: str = "Unknown"
    batting_team: str = "Unknown"
    bowling_team: str = "Unknown"
    innings: int = 1
    bowler: Optional[str] = None
    balls: List[Ball] = Field(..., min_length=1)
    match_state_overrides: dict = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/recommend")
def api_recommend(req: RecommendRequest):
    payload = req.model_dump()
    payload["balls"] = [b for b in payload["balls"]]
    try:
        result = recommend(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # keep the demo debuggable - surface the real error
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    n_signals = result["n_signals"]
    n_models_total = max(result["n_models_total"], 1)
    confidence = round(100 * n_signals / n_models_total)

    return {
        "role": result["role"],
        "phase": result["phase"],
        "text": result["text"],
        "chosen": result["chosen"],
        "confidence": confidence,
        "ranked_actions": result["ranked_actions"],
        "blocked": result["blocked"],
        "audit": result["audit"],
        "contributions": {
            action: [{"model_id": mid, "label": label, "contribution": contrib}
                      for mid, label, contrib in contribs]
            for action, contribs in result["contributions"].items()
        },
        "raw_model_outputs": result["raw_model_outputs"],
    }
