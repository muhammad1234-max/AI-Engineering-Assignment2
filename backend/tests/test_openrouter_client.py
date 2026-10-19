"""Comprehensive Unit Tests for OpenRouter Dual-API Client & Telemetry."""

import json
import os
from unittest.mock import patch

import httpx
import pytest

from backend.app.cube_engine import ALL_MOVES, SOLVED_STATE, Cube
from backend.app.openrouter_client import (
    MODEL_REGISTRY,
    OpenRouterClient,
    TargetModel,
    build_decision_payload,
    build_llm_payload,
    parse_decision_response,
    parse_llm_response,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Test Model Registry & Configuration
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Tests for model registration, category mapping, and environment overrides."""

    def test_all_four_target_models_registered(self):
        assert set(MODEL_REGISTRY.keys()) == {"jev", "luna", "clef", "claude"}

        jev = MODEL_REGISTRY["jev"]
        assert jev.default_id == "typesafe/jev-1.13"
        assert jev.category == "decision"
        assert jev.endpoint_path == "/api/alpha/decisions"

        luna = MODEL_REGISTRY["luna"]
        assert luna.default_id == "openai/gpt-6-luna-decisions"
        assert luna.category == "decision"
        assert luna.endpoint_path == "/api/alpha/decisions"

        clef = MODEL_REGISTRY["clef"]
        assert clef.default_id == "cloudflare/clef"
        assert clef.category == "decision"
        assert clef.endpoint_path == "/api/alpha/decisions"

        claude = MODEL_REGISTRY["claude"]
        assert claude.default_id == "anthropic/claude-opus-5.5"
        assert claude.category == "llm"
        assert claude.endpoint_path == "/api/v1/chat/completions"

    def test_model_id_env_override(self):
        with patch.dict(os.environ, {"MODEL_ID_JEV": "custom/jev-enterprise"}):
            target = resolve_model("jev")
            assert target.model_id == "custom/jev-enterprise"

    def test_resolve_by_key_and_by_id(self):
        # Resolve by key
        assert resolve_model("claude").key == "claude"
        # Resolve by default model ID
        assert resolve_model("anthropic/claude-opus-5.5").key == "claude"
        assert resolve_model("typesafe/jev-1.13").key == "jev"

    def test_resolve_unknown_model_raises_error(self):
        with pytest.raises(ValueError, match="Unknown target model"):
            resolve_model("unknown/nonexistent-model")


# ---------------------------------------------------------------------------
# Test Payload Builders
# ---------------------------------------------------------------------------

class TestPayloadBuilders:
    """Tests for payload structure adherence to specifications."""

    def test_decision_payload_structure(self):
        state = SOLVED_STATE
        recent_moves = ["R", "U'", "F2"]
        payload = build_decision_payload("typesafe/jev-1.13", state, recent_moves)

        assert payload["model"] == "typesafe/jev-1.13"
        assert payload["state"]["cube_facelets"] == state
        assert payload["state"]["recent_moves"] == ["R", "U'", "F2"]

        question = payload["questions"]["next_move"]
        assert question["type"] == "choice"
        assert "Choose the single next move" in question["instructions"]
        assert len(question["criteria"]) == 18
        for move in ALL_MOVES:
            assert move in question["criteria"]

    def test_llm_payload_structure(self):
        state = SOLVED_STATE
        recent_moves = ["L", "D2"]
        payload = build_llm_payload("anthropic/claude-opus-5.5", state, recent_moves)

        assert payload["model"] == "anthropic/claude-opus-5.5"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["temperature"] == 0.0

        messages = payload["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert '{"move": "<MOVE>"}' in messages[0]["content"]

        user_content = messages[1]["content"]
        assert state in user_content
        assert "L D2" in user_content
        for move in ALL_MOVES:
            assert move in user_content


# ---------------------------------------------------------------------------
# Test Response Parsers
# ---------------------------------------------------------------------------

class TestResponseParsers:
    """Tests for flexible Decision API parsing and strict LLM parsing."""

    def test_parse_decision_standard_string(self):
        data = {"answers": {"next_move": "R"}}
        move, conf, err = parse_decision_response(data)
        assert move == "R"
        assert conf is None
        assert err is None

    def test_parse_decision_object_with_confidence(self):
        data = {
            "answers": {
                "next_move": {
                    "choice": "U'",
                    "confidence": 0.92,
                }
            }
        }
        move, conf, err = parse_decision_response(data)
        assert move == "U'"
        assert conf == pytest.approx(0.92)
        assert err is None

    def test_parse_decision_alternative_decisions_key(self):
        data = {
            "decisions": {
                "next_move": {
                    "choice": "F2",
                    "probability": 0.85,
                }
            }
        }
        move, conf, err = parse_decision_response(data)
        assert move == "F2"
        assert conf == pytest.approx(0.85)
        assert err is None

    def test_parse_decision_missing_answer(self):
        data = {"status": "ok", "other_field": 123}
        move, conf, err = parse_decision_response(data)
        assert move is None
        assert "Could not find" in err

    def test_parse_llm_standard_json(self):
        content = json.dumps({"move": "R'"})
        data = {"choices": [{"message": {"content": content}}]}
        move, conf, err = parse_llm_response(data)
        assert move == "R'"
        assert conf is None
        assert err is None

    def test_parse_llm_markdown_fenced_json(self):
        content = "```json\n{\n  \"move\": \"B2\"\n}\n```"
        data = {"choices": [{"message": {"content": content}}]}
        move, conf, err = parse_llm_response(data)
        assert move == "B2"
        assert err is None

    def test_parse_llm_malformed_non_json(self):
        content = "I recommend moving the Right face clockwise."
        data = {"choices": [{"message": {"content": content}}]}
        move, conf, err = parse_llm_response(data)
        assert move is None
        assert "Failed to parse content as JSON" in err

    def test_parse_llm_missing_move_key(self):
        content = json.dumps({"reasoning": "Turning right", "step": 1})
        data = {"choices": [{"message": {"content": content}}]}
        move, conf, err = parse_llm_response(data)
        assert move is None
        assert "missing required 'move' key" in err


# ---------------------------------------------------------------------------
# Test HTTP Client Mocking, Retries, and Telemetry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOpenRouterClientHTTP:
    """Async HTTP mock tests for Decision and LLM adapters."""

    async def test_decision_model_success_http(self):
        called_url = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            called_url.append(str(request.url))
            req_body = json.loads(request.content)
            assert req_body["model"] == "typesafe/jev-1.13"
            return httpx.Response(
                200,
                json={"answers": {"next_move": {"choice": "R", "confidence": 0.95}}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        router_client = OpenRouterClient(
            api_key="test-key",
            mock_mode=False,
            http_client=client,
        )

        res = await router_client.get_next_move("jev", SOLVED_STATE, ["U"])
        assert res.is_valid is True
        assert res.move == "R"
        assert res.confidence == pytest.approx(0.95)
        assert res.retry_count == 0
        assert res.malformed_count == 0
        assert res.latency_ms > 0
        assert "/api/alpha/decisions" in called_url[0]

    async def test_llm_model_success_http(self):
        called_url = []

        def mock_handler(request: httpx.Request) -> httpx.Response:
            called_url.append(str(request.url))
            req_body = json.loads(request.content)
            assert req_body["model"] == "anthropic/claude-opus-5.5"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"move": "U\'"}',
                            }
                        }
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        router_client = OpenRouterClient(
            api_key="test-key",
            mock_mode=False,
            http_client=client,
        )

        res = await router_client.get_next_move("claude", SOLVED_STATE)
        assert res.is_valid is True
        assert res.move == "U'"
        assert res.retry_count == 0
        assert res.malformed_count == 0
        assert "/api/v1/chat/completions" in called_url[0]

    async def test_llm_malformed_retry_and_recovery(self):
        """Test that malformed JSON is logged, malformed_count incremented, and recovers on retry."""
        attempt = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                # First attempt: malformed free-form text instead of JSON
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Turn Right clockwise",
                                }
                            }
                        ]
                    },
                )
            else:
                # Second attempt: recovers with valid JSON
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"move": "R"}',
                                }
                            }
                        ]
                    },
                )

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        router_client = OpenRouterClient(
            api_key="test-key",
            mock_mode=False,
            max_retries=2,
            retry_delay_base=0.001,
            http_client=client,
        )

        res = await router_client.get_next_move("claude", SOLVED_STATE)
        assert res.is_valid is True
        assert res.move == "R"
        assert res.retry_count == 1
        assert res.malformed_count == 1
        assert attempt == 2

    async def test_invalid_move_string_rejection(self):
        """Test that invalid move strings (not in 18 Singmaster moves) are rejected and counted."""
        attempts = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            # Returns an illegal move ("X")
            return httpx.Response(
                200,
                json={"answers": {"next_move": "X"}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        router_client = OpenRouterClient(
            api_key="test-key",
            mock_mode=False,
            max_retries=2,
            retry_delay_base=0.001,
            http_client=client,
        )

        res = await router_client.get_next_move("jev", SOLVED_STATE)
        assert res.is_valid is False
        assert res.move is None
        assert res.retry_count == 2
        assert res.malformed_count == 3
        assert attempts == 3
        assert "Invalid move 'X'" in res.error_message

    async def test_http_429_rate_limit_retry(self):
        attempts = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, text="Rate limit exceeded")
            return httpx.Response(
                200,
                json={"answers": {"next_move": "F"}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        router_client = OpenRouterClient(
            api_key="test-key",
            mock_mode=False,
            max_retries=2,
            retry_delay_base=0.001,
            http_client=client,
        )

        res = await router_client.get_next_move("clef", SOLVED_STATE)
        assert res.is_valid is True
        assert res.move == "F"
        assert res.retry_count == 1
        assert res.malformed_count == 0

    async def test_http_timeout_retry(self):
        attempts = 0

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ReadTimeout("Connection timed out")
            return httpx.Response(
                200,
                json={"answers": {"next_move": "D2"}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        router_client = OpenRouterClient(
            api_key="test-key",
            mock_mode=False,
            max_retries=2,
            retry_delay_base=0.001,
            http_client=client,
        )

        res = await router_client.get_next_move("luna", SOLVED_STATE)
        assert res.is_valid is True
        assert res.move == "D2"
        assert res.retry_count == 1


# ---------------------------------------------------------------------------
# Test Offline Mock / Dry-Run Mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOfflineMockMode:
    """Tests for offline mock / dry-run simulator."""

    async def test_mock_mode_decision_model(self):
        cube = Cube()
        cube.apply_move("R")  # Scrambled by 1 move

        client = OpenRouterClient(mock_mode=True)
        res = await client.get_next_move("jev", cube.to_facelet_string())

        assert res.is_valid is True
        assert res.move == "R'"  # Optimal BFS move solves it
        assert res.confidence is not None
        assert 0.80 <= res.confidence <= 1.0
        assert res.model_category == "decision"
        assert res.retry_count == 0
        assert res.malformed_count == 0

    async def test_mock_mode_llm_model(self):
        cube = Cube()
        cube.apply_moves("U F")

        client = OpenRouterClient(mock_mode=True)
        res = await client.get_next_move("claude", cube.to_facelet_string())

        assert res.is_valid is True
        assert res.move in ALL_MOVES
        assert res.model_category == "llm"
        assert res.retry_count == 0
        assert res.malformed_count == 0
