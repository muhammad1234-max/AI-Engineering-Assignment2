"""Unit and Integration Tests for Benchmark Runner, Race Engine, and FastAPI Endpoints."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from backend.app.benchmark_runner import (
    DATA_DIR,
    MAX_RACE_MOVES,
    RESULTS_FILE,
    BenchmarkRunner,
    BenchmarkSuiteResult,
    ModelRaceResult,
    RaceResult,
    compute_suite_aggregate_stats,
    generate_assignment_analysis,
    load_benchmark_results,
    run_benchmark_suite,
)
from backend.app.cube_engine import Cube
from backend.app.main import app
from backend.app.openrouter_client import MODEL_REGISTRY, OpenRouterClient


# ---------------------------------------------------------------------------
# Test Benchmark Runner Engine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBenchmarkRunner:
    """Tests for single-puzzle race engine and scramble preparation."""

    async def test_prepare_scramble_warmup(self):
        runner = BenchmarkRunner()
        scramble, cube, dist, moves = runner.prepare_scramble(tier="warmup", seed=10)
        assert len(scramble) == 3
        assert len(cube.to_facelet_string()) == 54
        assert dist <= 3
        assert len(moves) == dist

    async def test_prepare_scramble_explicit(self):
        runner = BenchmarkRunner()
        scramble, cube, dist, moves = runner.prepare_scramble(scramble_moves=["R", "U", "R'"])
        assert scramble == ["R", "U", "R'"]
        assert dist == 3
        assert len(moves) == dist

    async def test_run_race_four_models_mock(self):
        """Verify that a race executes all 4 models, calculates ranks, and returns valid data."""
        client = OpenRouterClient(mock_mode=True)
        runner = BenchmarkRunner(client=client, max_moves=10)

        result: RaceResult = await runner.run_race(
            scramble_moves=["R", "U'"],
            tier="primary",
            seed=42,
        )

        assert result.race_id is not None
        assert result.shortest_distance == 2
        assert len(result.participants) == 4
        assert set(result.participants.keys()) == {"jev", "luna", "clef", "claude"}

        ranks = [p.rank for p in result.participants.values()]
        assert sorted(ranks) == [1, 2, 3, 4], "Ranks must be exactly 1, 2, 3, 4 without duplicates"

        for p in result.participants.values():
            assert p.total_moves > 0
            assert p.status in ("SOLVED", "RAN_OUT_OF_MOVES")
            assert len(p.steps) == p.total_moves
            assert p.final_facelets is not None
            if p.status == "SOLVED":
                assert p.optimality_gap is not None
                assert p.optimality_gap >= 0

    async def test_race_max_moves_cutoff(self):
        """Verify that models stop immediately when max_moves is reached."""
        client = OpenRouterClient(mock_mode=True)
        # Deep scramble with max_moves cutoff = 2
        runner = BenchmarkRunner(client=client, max_moves=2)

        result: RaceResult = await runner.run_race(
            scramble_moves=["R", "U", "F", "D", "L", "B"],
            tier="primary",
        )

        for p in result.participants.values():
            assert p.total_moves <= 2
            if not p.is_solved:
                assert p.status == "RAN_OUT_OF_MOVES"

    async def test_stream_race_events_sse(self):
        """Verify that stream_race_events yields valid SSE messages."""
        client = OpenRouterClient(mock_mode=True)
        runner = BenchmarkRunner(client=client, max_moves=4)

        events_received = []
        async for chunk in runner.stream_race_events(scramble_moves=["R"]):
            events_received.append(chunk)

        full_stream = "".join(events_received)
        assert "event: race_start" in full_stream
        assert "event: move_step" in full_stream
        assert "event: race_complete" in full_stream


# ---------------------------------------------------------------------------
# Test Statistical Aggregator & Assignment Analysis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStatisticalAggregator:
    """Tests for multi-puzzle statistics and automated assignment analysis."""

    async def test_multi_puzzle_benchmark_suite(self, tmp_path):
        client = OpenRouterClient(mock_mode=True)
        runner = BenchmarkRunner(client=client, max_moves=8)

        # Run 2 puzzles
        suite_res: BenchmarkSuiteResult = await run_benchmark_suite(
            num_puzzles=2,
            tier="primary",
            seed=10,
            runner=runner,
        )

        assert suite_res.num_puzzles == 2
        assert len(suite_res.puzzles) == 2
        assert len(suite_res.model_stats) == 4

        for key, stats in suite_res.model_stats.items():
            assert stats.total_trials == 2
            assert 0.0 <= stats.success_rate_pct <= 100.0
            assert stats.avg_moves > 0.0
            assert stats.avg_latency_ms > 0.0
            assert stats.total_malformed == 0  # Under mock mode
            # Finishing positions sum to total trials
            finishes = (
                stats.first_place_finishes
                + stats.second_place_finishes
                + stats.third_place_finishes
                + stats.fourth_place_finishes
            )
            assert finishes == 2

        # Check automated analysis
        analysis = suite_res.analysis_insights
        assert "question_1_system1_vs_system2" in analysis
        assert "question_2_optimality_gap_and_success_rate" in analysis
        assert "question_3_malformed_response_rates" in analysis
        assert "question_4_exact_bfs_scalability" in analysis
        assert "question_5_robotics_deployment_recommendations" in analysis

    async def test_compute_suite_aggregate_stats_math(self):
        """Verify mathematical formulas for success rate, solve time, and optimality gap."""
        # Build mock puzzle results
        participants = {
            "jev": ModelRaceResult(
                model_key="jev",
                model_name="TypeSafe JEV",
                model_id="typesafe/jev-1.13",
                category="decision",
                rank=1,
                status="SOLVED",
                total_moves=4,
                total_time_ms=200.0,
                avg_latency_ms=50.0,
                optimality_gap=1,  # shortest was 3
                total_malformed=0,
                total_retries=0,
                is_solved=True,
                final_facelets="",
            ),
            "claude": ModelRaceResult(
                model_key="claude",
                model_name="Claude Opus",
                model_id="anthropic/claude-opus-5.5",
                category="llm",
                rank=2,
                status="RAN_OUT_OF_MOVES",
                total_moves=30,
                total_time_ms=30000.0,
                avg_latency_ms=1000.0,
                optimality_gap=None,  # Not solved
                total_malformed=1,
                total_retries=1,
                is_solved=False,
                final_facelets="",
            ),
        }

        mock_puzzle = RaceResult(
            race_id="test1",
            tier="primary",
            scramble=["R", "U", "F"],
            initial_state="",
            shortest_distance=3,
            shortest_moves=["F'", "U'", "R'"],
            participants=participants,
            timestamp="",
        )

        stats = compute_suite_aggregate_stats([mock_puzzle])

        jev_stats = stats["jev"]
        assert jev_stats.total_trials == 1
        assert jev_stats.solved_trials == 1
        assert jev_stats.success_rate_pct == 100.0
        assert jev_stats.avg_optimality_gap == 1.0
        assert jev_stats.first_place_finishes == 1

        claude_stats = stats["claude"]
        assert claude_stats.total_trials == 1
        assert claude_stats.solved_trials == 0
        assert claude_stats.success_rate_pct == 0.0
        assert claude_stats.avg_optimality_gap is None
        assert claude_stats.total_malformed == 1


# ---------------------------------------------------------------------------
# Test FastAPI Endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFastAPIEndpoints:
    """Integration tests for FastAPI REST and SSE endpoints."""

    async def test_get_health_endpoint(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "healthy"
            assert "mock_mode" in data
            assert data["models_count"] == 4

    async def test_get_models_endpoint(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/models")
            assert res.status_code == 200
            data = res.json()
            assert len(data) == 4
            keys = [m["key"] for m in data]
            assert set(keys) == {"jev", "luna", "clef", "claude"}

    async def test_post_scramble_endpoint(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/api/scramble", json={"tier": "primary", "seed": 42})
            assert res.status_code == 200
            data = res.json()
            assert data["tier"] == "primary"
            assert len(data["scramble"]) == 4
            assert len(data["initial_state"]) == 54
            assert data["shortest_distance"] > 0
            assert len(data["shortest_moves"]) == data["shortest_distance"]

    async def test_post_race_run_endpoint(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(
                "/api/race/run",
                json={"scramble": ["R", "U'"], "max_moves": 10},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["shortest_distance"] == 2
            assert len(data["participants"]) == 4
            for participant in data["participants"].values():
                assert participant["rank"] in (1, 2, 3, 4)
                assert participant["total_moves"] > 0

    async def test_post_race_stream_endpoint(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "POST",
                "/api/race/stream",
                json={"scramble": ["U"], "max_moves": 4},
            ) as stream_res:
                assert stream_res.status_code == 200
                chunks = []
                async for chunk in stream_res.aiter_text():
                    chunks.append(chunk)

                full_text = "".join(chunks)
                assert "event: race_start" in full_text
                assert "event: move_step" in full_text
                assert "event: race_complete" in full_text

    async def test_post_benchmark_suite_endpoint(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(
                "/api/benchmark/suite",
                json={"num_puzzles": 1, "tier": "warmup", "seed": 99},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["num_puzzles"] == 1
            assert len(data["puzzles"]) == 1
            assert len(data["model_stats"]) == 4

    async def test_get_benchmark_stats_endpoint(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/benchmark/stats")
            assert res.status_code == 200
            data = res.json()
            assert "model_stats" in data
            assert "analysis_insights" in data
            assert len(data["model_stats"]) == 4

    async def test_serve_frontend_index_and_static(self):
        """Verify frontend index.html and static assets are served properly."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Root index.html
            res = await client.get("/")
            assert res.status_code == 200
            assert "text/html" in res.headers.get("content-type", "")
            # Check mandatory student banner in served HTML
            assert "Muhammad Abbas" in res.text
            assert "23K-0068" in res.text
            assert "student-identity-banner" in res.text

            # 2. Static CSS
            res_css = await client.get("/static/css/styles.css")
            assert res_css.status_code == 200
            assert "text/css" in res_css.headers.get("content-type", "")

            # 3. Static JS
            res_js = await client.get("/static/js/app.js")
            assert res_js.status_code == 200
            assert "javascript" in res_js.headers.get("content-type", "")

    async def test_abort_race_endpoint(self):
        """Test POST /api/race/abort cancels or returns abort status."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/api/race/abort")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "aborted"
            assert "message" in data

    async def test_config_endpoint_and_budget_tracking(self):
        """Test POST /api/config runtime switching and budget tracking."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(
                "/api/config",
                json={"mock_mode": True, "reset_budget": True},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["mock_mode"] is True
            assert data["live_api_calls_used"] == 0
            assert data["max_live_calls"] >= 1

    async def test_cost_guardrail_blocks_when_limit_reached(self):
        """Test that reaching MAX_LIVE_API_CALLS_PER_SESSION blocks further live calls."""
        from backend.app.openrouter_client import OpenRouterClient
        from backend.app.cube_engine import Cube

        # Client with max 2 calls in live mode
        mock_client = OpenRouterClient(api_key="mock-key", mock_mode=False, max_live_calls=2)
        mock_client.live_api_calls_used = 2  # Already reached limit

        state = Cube().to_facelet_string()
        res = await mock_client.get_next_move("jev", state)
        assert res.is_valid is False
        assert "Hard Cost Guardrail" in (res.error_message or "")
        assert mock_client.live_api_calls_used == 2  # Did not increment or make HTTP call




