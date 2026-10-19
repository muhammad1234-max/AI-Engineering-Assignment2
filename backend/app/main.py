"""FastAPI Application for AI Rubik's Cube Race Benchmark.

Exposes REST and Server-Sent Event (SSE) endpoints:
- GET  /api/health: Service status & configuration
- GET  /api/models: Registered benchmark target models
- POST /api/scramble: Generate scrambles with exact BFS shortest-path solutions
- POST /api/race/run: Execute a full 4-model race synchronously
- POST /api/race/stream: Live SSE stream of move-by-move race progress
- POST /api/benchmark/suite: Run multi-puzzle benchmark suite
- GET  /api/benchmark/stats: Retrieve latest statistical aggregates & analysis
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkSuiteResult,
    RaceResult,
    compute_suite_aggregate_stats,
    generate_assignment_analysis,
    load_benchmark_results,
    run_benchmark_suite,
)
from backend.app.cube_engine import Cube, solve_bfs
from backend.app.openrouter_client import (
    MODEL_REGISTRY,
    OPENROUTER_BASE_URL,
    OpenRouterClient,
)

# Initialize FastAPI application
app = FastAPI(
    title="AI Rubik's Cube Race Benchmark API",
    description="Live benchmark comparing Decision models vs. Frontier LLMs on 3D Rubik's Cube solving.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global benchmark runner instance
runner = BenchmarkRunner()

# Path to frontend directory and static assets
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["Frontend"])
async def serve_index():
    """Serve the Rubik's Cube Race web application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=404,
        content={"message": "Frontend index.html not found"},
    )


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class ScrambleRequest(BaseModel):
    tier: str = Field(
        default="primary",
        description="Scramble difficulty: 'warmup' (3 moves), 'primary' (4-5 moves), or 'extension' (15-20 moves)",
    )
    seed: Optional[int] = Field(default=None, description="Optional seed for deterministic generation")
    length: Optional[int] = Field(default=None, description="Explicit scramble length override")


class RaceRunRequest(BaseModel):
    scramble: Optional[List[str]] = Field(default=None, description="Explicit sequence of moves to apply")
    tier: str = Field(default="primary", description="Tier if scramble is not provided")
    seed: Optional[int] = Field(default=None, description="Seed for scramble generation")
    length: Optional[int] = Field(default=None, description="Explicit scramble length")
    max_moves: int = Field(default=60, description="Max turn cutoff for each model (default 60)")


class BenchmarkSuiteRequest(BaseModel):
    num_puzzles: int = Field(default=5, ge=1, le=20, description="Number of benchmark puzzles (default 5)")
    tier: str = Field(default="primary", description="Benchmark tier")
    seed: int = Field(default=42, description="Base seed for reproducibility")


class ConfigUpdateRequest(BaseModel):
    api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    mock_mode: Optional[bool] = Field(default=None, description="Set to true for simulated mock mode, false for live OpenRouter API calls")
    reset_budget: Optional[bool] = Field(default=False, description="Reset live API call counter to zero")
    max_live_calls: Optional[int] = Field(default=None, description="Set maximum live API call limit")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Return health status, current environment, mock/live configuration, and API budget."""
    mock_mode = runner.client.mock_mode
    api_key_set = bool(runner.client.api_key)
    return {
        "status": "healthy",
        "mock_mode": mock_mode,
        "api_key_configured": api_key_set,
        "live_api_calls_used": runner.client.live_api_calls_used,
        "max_live_calls": runner.client.max_live_calls,
        "base_url": OPENROUTER_BASE_URL,
        "models_count": len(MODEL_REGISTRY),
        "environment": os.getenv("ENVIRONMENT", "development"),
    }


@app.post("/api/config", tags=["System"])
async def update_config(req: ConfigUpdateRequest) -> Dict[str, Any]:
    """Update OpenRouter API key, toggle mock/live mode, or reset budget counter at runtime."""
    if req.api_key is not None:
        runner.client.api_key = req.api_key.strip()
    if req.mock_mode is not None:
        runner.client.mock_mode = req.mock_mode
    if req.reset_budget:
        runner.client.live_api_calls_used = 0
    if req.max_live_calls is not None and req.max_live_calls > 0:
        runner.client.max_live_calls = req.max_live_calls
    return {
        "status": "updated",
        "mock_mode": runner.client.mock_mode,
        "api_key_configured": bool(runner.client.api_key),
        "live_api_calls_used": runner.client.live_api_calls_used,
        "max_live_calls": runner.client.max_live_calls,
    }


@app.post("/api/mode/toggle", tags=["System"])
async def toggle_execution_mode() -> Dict[str, Any]:
    """Toggle between Simulated Mock Mode ($0 Cost) and Live OpenRouter API."""
    runner.client.mock_mode = not runner.client.mock_mode
    return {
        "status": "toggled",
        "mock_mode": runner.client.mock_mode,
        "mode_label": "Simulated / $0 Cost" if runner.client.mock_mode else "Live OpenRouter API",
        "api_key_configured": bool(runner.client.api_key),
        "live_api_calls_used": runner.client.live_api_calls_used,
        "max_live_calls": runner.client.max_live_calls,
    }




@app.get("/api/models", tags=["Models"])
async def get_models() -> List[Dict[str, Any]]:
    """Return the 4 target models configured for the benchmark race."""
    models_info = []
    for key, target in MODEL_REGISTRY.items():
        models_info.append(
            {
                "key": key,
                "name": target.name,
                "model_id": target.model_id,
                "category": target.category,
                "endpoint_path": target.endpoint_path,
                "full_endpoint": target.full_endpoint,
            }
        )
    return models_info


@app.post("/api/scramble", tags=["Cube"])
async def generate_cube_scramble(req: ScrambleRequest) -> Dict[str, Any]:
    """Generate a valid non-redundant scramble and calculate exact BFS shortest path."""
    scramble, cube, shortest_dist, shortest_moves = runner.prepare_scramble(
        tier=req.tier, seed=req.seed, length=req.length
    )
    return {
        "tier": req.tier,
        "length": len(scramble),
        "scramble": scramble,
        "initial_state": cube.to_facelet_string(),
        "shortest_distance": shortest_dist,
        "shortest_moves": shortest_moves,
    }


@app.post("/api/race/run", tags=["Race"])
async def run_race_synchronous(req: RaceRunRequest) -> Dict[str, Any]:
    """Execute a 4-model race concurrently and return complete final results."""
    runner.max_moves = req.max_moves
    result: RaceResult = await runner.run_race(
        scramble_moves=req.scramble,
        tier=req.tier,
        seed=req.seed,
        length=req.length,
    )
    return result.to_dict()


@app.post("/api/race/stream", tags=["Race"])
async def run_race_sse_stream(req: RaceRunRequest):
    """Execute a 4-model race streaming real-time move-by-move events via SSE."""
    runner.max_moves = req.max_moves
    event_generator = runner.stream_race_events(
        scramble_moves=req.scramble,
        tier=req.tier,
        seed=req.seed,
        length=req.length,
    )
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/race/abort", tags=["Race"])
async def abort_race() -> Dict[str, Any]:
    """Immediately abort any active in-flight race and cancel background tasks."""
    aborted = runner.abort_current_race()
    return {
        "status": "aborted",
        "was_running": aborted,
        "message": "Race execution successfully aborted.",
    }



@app.post("/api/benchmark/suite", tags=["Benchmark"])
async def execute_benchmark_suite(req: BenchmarkSuiteRequest) -> Dict[str, Any]:
    """Run a multi-puzzle benchmark suite across all 4 models and save to disk."""
    suite_res: BenchmarkSuiteResult = await run_benchmark_suite(
        num_puzzles=req.num_puzzles,
        tier=req.tier,
        seed=req.seed,
        runner=runner,
    )
    return suite_res.to_dict()


@app.get("/api/benchmark/stats", tags=["Benchmark"])
async def get_benchmark_statistics() -> Dict[str, Any]:
    """Retrieve cumulative benchmark stats and automated Section 7 assignment analysis."""
    cached = load_benchmark_results()
    if cached is not None:
        return cached

    # If no benchmark run has been stored yet, generate an initial 5-puzzle benchmark
    suite_res = await run_benchmark_suite(num_puzzles=5, tier="primary", seed=42, runner=runner)
    return suite_res.to_dict()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)
