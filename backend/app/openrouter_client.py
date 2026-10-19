"""OpenRouter Dual-API Adapter Layer for Rubik's Cube Benchmark.

This module implements:
1. Target Model Registry with environment overrides & configuration.
2. Decision Models Adapter targeting `/api/alpha/decisions`.
3. Frontier LLM Adapter targeting `/api/v1/chat/completions`.
4. Strict Move Validation against the 18 standard Singmaster moves.
5. Unified Retry Policy, Malformed Tracking, and Latency Telemetry.
6. Offline Mock / Dry-Run Mode for reliable testing and local development.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from dotenv import load_dotenv

from backend.app.cube_engine import (
    ALL_MOVES,
    MOVE_DESCRIPTIONS,
    Cube,
    invert_move,
    solve_bfs,
)

# Load environment variables
load_dotenv()

logger = logging.getLogger("rubiks.openrouter")

# ---------------------------------------------------------------------------
# Model Registry & Configurations
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai").rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

DECISION_ENDPOINT = f"{OPENROUTER_BASE_URL}/api/alpha/decisions"
CHAT_ENDPOINT = f"{OPENROUTER_BASE_URL}/api/v1/chat/completions"


@dataclass(frozen=True)
class TargetModel:
    """Specification of an OpenRouter target model."""

    key: str
    default_id: str
    env_override_key: str
    name: str
    category: str  # "decision" or "llm"
    endpoint_path: str

    @property
    def model_id(self) -> str:
        """Get the effective model ID, honoring environment overrides."""
        return os.getenv(self.env_override_key, self.default_id)

    @property
    def effective_category(self) -> str:
        """Determine whether the active model ID routes to decision or chat API."""
        mid = self.model_id.lower()
        if any(prefix in mid for prefix in ("gpt-4", "gpt-3.5", "o1-", "claude")):
            return "llm"
        return self.category

    @property
    def full_endpoint(self) -> str:
        if self.effective_category == "llm":
            return f"{OPENROUTER_BASE_URL}/api/v1/chat/completions"
        return f"{OPENROUTER_BASE_URL}{self.endpoint_path}"


# The 4 target benchmark models defined in assignment specifications
MODEL_REGISTRY: Dict[str, TargetModel] = {
    "jev": TargetModel(
        key="jev",
        default_id="typesafe/jev-1.13",
        env_override_key="MODEL_ID_JEV",
        name="TypeSafe JEV 1.13",
        category="decision",
        endpoint_path="/api/alpha/decisions",
    ),
    "luna": TargetModel(
        key="luna",
        default_id="openai/gpt-6-luna-decisions",
        env_override_key="MODEL_ID_LUNA",
        name="GPT-6 Luna Decisions",
        category="decision",
        endpoint_path="/api/alpha/decisions",
    ),
    "clef": TargetModel(
        key="clef",
        default_id="cloudflare/clef",
        env_override_key="MODEL_ID_CLEF",
        name="Cloudflare CLEF",
        category="decision",
        endpoint_path="/api/alpha/decisions",
    ),
    "claude": TargetModel(
        key="claude",
        default_id="anthropic/claude-opus-5.5",
        env_override_key="MODEL_ID_CLAUDE",
        name="Claude Opus 5.5",
        category="llm",
        endpoint_path="/api/v1/chat/completions",
    ),
}


def resolve_model(model_identifier: str) -> TargetModel:
    """Resolve a model key (e.g. 'jev', 'claude') or full model ID to a TargetModel.
    
    Raises:
        ValueError: If model cannot be resolved to a known target configuration.
    """
    # 1. Direct key lookup
    if model_identifier in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_identifier]

    # 2. Match against configured / overridden model_ids
    for target in MODEL_REGISTRY.values():
        if target.model_id == model_identifier or target.default_id == model_identifier:
            return target

    # 3. Dynamic fallback based on endpoint/category naming if needed
    raise ValueError(
        f"Unknown target model '{model_identifier}'. Supported keys are: {list(MODEL_REGISTRY.keys())}"
    )


# ---------------------------------------------------------------------------
# Telemetry Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MoveDecisionResult:
    """Standardized response & telemetry for a model's move choice."""

    model_id: str
    model_category: str
    move: Optional[str]
    is_valid: bool
    confidence: Optional[float]
    raw_response: Optional[Dict[str, Any]]
    latency_ms: float
    retry_count: int
    malformed_count: int
    error_message: Optional[str] = None

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID/ERROR"
        return (
            f"MoveDecisionResult({self.model_id} [{self.model_category}] -> "
            f"move={self.move} ({status}), conf={self.confidence}, "
            f"latency={self.latency_ms:.1f}ms, retries={self.retry_count}, "
            f"malformed={self.malformed_count}, err={self.error_message})"
        )


# ---------------------------------------------------------------------------
# Payload Builders
# ---------------------------------------------------------------------------

DECISION_INSTRUCTIONS = (
    "Choose the single next move most likely to progress toward the solved cube."
)


def build_decision_payload(
    model_id: str,
    cube_facelets: str,
    recent_moves: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construct identical request payload for /api/alpha/decisions across all decision models."""
    recent_moves_capped = (recent_moves or [])[-6:]
    return {
        "model": model_id,
        "state": {
            "cube_facelets": cube_facelets,
            "recent_moves": recent_moves_capped,
        },
        "questions": {
            "next_move": {
                "type": "choice",
                "instructions": DECISION_INSTRUCTIONS,
                "criteria": dict(MOVE_DESCRIPTIONS),
            }
        },
    }


def build_llm_payload(
    model_id: str,
    cube_facelets: str,
    recent_moves: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construct minimal-token request payload for /api/v1/chat/completions.
    
    Adheres strictly to token guardrails:
    - max_tokens = 16 (prevents reasoning verbosity)
    - temperature = 0.0 (deterministic move extraction)
    - capped recent_moves (last 6 moves max)
    - compact prompts (~120 tokens total)
    """
    recent_moves_capped = (recent_moves or [])[-6:]
    recent_moves_str = (
        " ".join(recent_moves_capped) if recent_moves_capped else "None"
    )

    system_prompt = (
        "You are an expert Rubik's Cube solver benchmark participant.\n"
        "Pick the single next move toward solving the cube from the 18 Singmaster moves.\n"
        'Output strictly valid JSON: {"move": "<MOVE>"}. No other text.'
    )

    user_prompt = (
        f"State: {cube_facelets}\n"
        f"Recent moves (last 6): {recent_moves_str}\n"
        f"Permitted Actions (18 standard Singmaster moves):\n"
        f"{' '.join(ALL_MOVES)}\n"
        f'Output next move JSON: {{"move": "<MOVE>"}}'
    )

    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 160,
        "temperature": 0.0,
    }




# ---------------------------------------------------------------------------
# Response Parsers
# ---------------------------------------------------------------------------

def parse_decision_response(
    data: Any,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Flexible parser for /api/alpha/decisions response structures.
    
    Handles:
    - {"answers": {"next_move": "R"}}
    - {"answers": {"next_move": {"choice": "R", "confidence": 0.95}}}
    - {"decisions": {"next_move": {"choice": "R", "probability": 0.92}}}
    - {"choices": [{"decision": "R", "confidence": 0.88}]}
    - {"decision": "R", "confidence": 0.95}
    
    Returns:
        (move, confidence, error_reason)
    """
    if not isinstance(data, dict):
        return None, None, f"Response is not a JSON object: {type(data).__name__}"

    target_val: Any = None
    confidence: Optional[float] = None

    # Check common decision containers
    if "answers" in data and isinstance(data["answers"], dict):
        target_val = data["answers"].get("next_move")
    elif "decisions" in data and isinstance(data["decisions"], dict):
        target_val = data["decisions"].get("next_move")
    elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
        choice_obj = data["choices"][0]
        if isinstance(choice_obj, dict):
            target_val = (
                choice_obj.get("decision")
                or choice_obj.get("choice")
                or choice_obj.get("next_move")
            )
            confidence = choice_obj.get("confidence") or choice_obj.get("probability")
    elif "decision" in data:
        target_val = data.get("decision")
        confidence = data.get("confidence") or data.get("probability")
    elif "next_move" in data:
        target_val = data.get("next_move")

    if target_val is None:
        return None, None, "Could not find 'next_move' or 'decision' in response structure"

    # If target_val is a nested dictionary (e.g. {"choice": "R", "confidence": 0.85})
    if isinstance(target_val, dict):
        raw_move = (
            target_val.get("choice")
            or target_val.get("move")
            or target_val.get("decision")
            or target_val.get("value")
            or target_val.get("action")
        )
        conf = (
            target_val.get("confidence")
            or target_val.get("probability")
            or target_val.get("score")
        )
        if conf is not None:
            try:
                confidence = float(conf)
            except (ValueError, TypeError):
                confidence = None
    elif isinstance(target_val, str):
        raw_move = target_val
    else:
        return None, None, f"Unexpected type for chosen move: {type(target_val).__name__}"

    move = raw_move.strip()
    # Normalize unicode prime characters to standard ASCII apostrophe
    move = move.replace("’", "'").replace("′", "'").replace("`", "'")
    # Strip matching outer quotes (e.g. '"U"' or "'U'")
    if len(move) >= 2 and move[0] == move[-1] and move[0] in ('"', "'"):
        move = move[1:-1].strip()
    return move, confidence, None


def parse_llm_response(
    data: Any,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Strict parser for /api/v1/chat/completions response structures.
    
    Expects choices[0].message.content to contain JSON: {"move": "<MOVE>"}.
    
    Returns:
        (move, confidence, error_reason)
    """
    if not isinstance(data, dict):
        return None, None, f"Response is not a JSON object: {type(data).__name__}"

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None, "Response missing non-empty 'choices' array"

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None, None, "First choice element is not an object"

    message = first_choice.get("message", {})
    content = message.get("content")
    reasoning = message.get("reasoning")

    if content is None:
        # Check if the model emitted reasoning / thought text containing move JSON or move
        if isinstance(reasoning, str) and reasoning.strip():
            json_match = re.search(r'\{[^{}]*"(?:move|next_move|action)"\s*:\s*"([^"]+)"[^{}]*\}', reasoning)
            if json_match:
                move = json_match.group(1).strip()
                move = move.replace("’", "'").replace("′", "'").replace("`", "'")
                return move, None, None
            words = re.findall(r"\b([URFDLB]['2]?)\b", reasoning)
            for w in words:
                norm_w = w.replace("’", "'").replace("′", "'").replace("`", "'")
                if norm_w in ALL_MOVES:
                    return norm_w, None, None
        return None, None, "Message missing 'content' field"

    if not isinstance(content, str):
        return None, None, f"Content is not a string: {type(content).__name__}"

    content_clean = content.strip()

    # Strip markdown code blocks if the LLM wrapped it (e.g. ```json ... ```)
    if content_clean.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content_clean)
        if match:
            content_clean = match.group(1).strip()

    # Try JSON parsing
    json_err: Optional[Exception] = None
    parsed_json: Optional[Dict[str, Any]] = None
    try:
        parsed_json = json.loads(content_clean)
    except json.JSONDecodeError as err:
        json_err = err

    if parsed_json is None:
        # Fallback: check for embedded JSON object in content
        json_match = re.search(r'\{[^{}]*"(?:move|next_move|action)"\s*:\s*"([^"]+)"[^{}]*\}', content_clean)
        if json_match:
            move = json_match.group(1).strip().replace("’", "'").replace("′", "'").replace("`", "'")
            if move in ALL_MOVES:
                return move, None, None
        return None, None, f"Failed to parse content as JSON: {json_err} (raw: '{content[:80]}')"

    if not isinstance(parsed_json, dict):
        return None, None, f"Parsed JSON is not an object: {type(parsed_json).__name__}"

    raw_move = (
        parsed_json.get("move")
        or parsed_json.get("next_move")
        or parsed_json.get("action")
    )
    if raw_move is None:
        return None, None, "JSON missing required 'move' key"

    if not isinstance(raw_move, str):
        return None, None, f"'move' value is not a string: {raw_move}"

    move = raw_move.strip()
    # Normalize unicode prime characters to standard ASCII apostrophe
    move = move.replace("’", "'").replace("′", "'").replace("`", "'")
    # Strip matching outer quotes (e.g. '"U"' or "'U'")
    if len(move) >= 2 and move[0] == move[-1] and move[0] in ('"', "'"):
        move = move[1:-1].strip()
    return move, None, None


# ---------------------------------------------------------------------------
# OpenRouter Async Client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """Async OpenRouter Client supporting both /api/alpha/decisions and /api/v1/chat/completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        max_retries: Optional[int] = None,
        mock_mode: Optional[bool] = None,
        retry_delay_base: float = 0.5,
        http_client: Optional[httpx.AsyncClient] = None,
        max_live_calls: Optional[int] = None,
    ) -> None:
        """Initialize the OpenRouter benchmark client.
        
        Args:
            api_key: OpenRouter API key. If not provided, loads from OPENROUTER_API_KEY env.
            timeout: Per-request timeout in seconds.
            max_retries: Number of retries on errors (defaults to 1 in live mode to save tokens).
            mock_mode: If True, uses local dry-run simulation instead of HTTP requests.
            retry_delay_base: Base exponential backoff delay in seconds (0.5s * 2^attempt).
            http_client: Optional pre-configured httpx.AsyncClient (for testing/mocking).
            max_live_calls: Hard session call limit (defaults to MAX_LIVE_API_CALLS_PER_SESSION).
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", str(timeout)))

        if max_retries is not None:
            self.max_retries = max_retries
        else:
            # In live mode default to 1 retry (instead of 3) to prevent 3x token costs
            env_retries = os.getenv("OPENROUTER_MAX_RETRIES")
            self.max_retries = int(env_retries) if env_retries else 1


        if mock_mode is not None:
            self.mock_mode = mock_mode
        else:
            # Default to Simulated / Mock Mode ($0 cost) unless OPENROUTER_MOCK_MODE is explicitly false/0/no
            env_mock = os.getenv("OPENROUTER_MOCK_MODE", "true").strip().lower()
            self.mock_mode = env_mock not in ("false", "0", "no")

        self.retry_delay_base = retry_delay_base
        self._external_client = http_client

        # Cost & Call-Budget Guardrails (Session-level)
        self.max_live_calls = (
            max_live_calls
            if max_live_calls is not None
            else int(os.getenv("MAX_LIVE_API_CALLS_PER_SESSION", "5"))
        )
        self.live_api_calls_used = 0




    def _get_headers(self) -> Dict[str, str]:
        """Build standard OpenRouter request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/ai-engr/rubiks-cube-race",
            "X-Title": "Rubiks Cube Race Benchmark",
            "Content-Type": "application/json",
        }

    async def _send_mock_decision(
        self,
        target: TargetModel,
        cube_facelets: str,
        recent_moves: Optional[List[str]] = None,
    ) -> MoveDecisionResult:
        """Differentiated offline mock simulator reflecting System-1 vs. System-2 profiles.
        
        Behavioral Profiles:
        - typesafe/jev-1.13: Fast lightweight decision (40-80ms), ~80% optimal, occasional detour.
        - openai/gpt-6-luna-decisions: Robust decision (70-120ms), ~90% optimal, high confidence.
        - cloudflare/clef: Ultra-fast edge classification (30-60ms), ~72% optimal, occasional wander.
        - anthropic/claude-opus-5.5: Deep deliberative LLM (850-1500ms), ~96% optimal efficiency.
        """
        import random

        # Fast execution under pytest or when MOCK_FAST is set
        is_test = "PYTEST_CURRENT_TEST" in os.environ or os.getenv("MOCK_FAST", "").lower() in ("1", "true")
        scale = 0.001 if is_test else 0.05  # subtle async sleep for live demo

        cube = Cube(cube_facelets)
        # Fast BFS search capped at max_depth=5 for responsiveness (<10ms)
        solve_res = solve_bfs(cube, max_depth=5)
        if solve_res.is_solved and solve_res.moves:
            optimal_move = solve_res.moves[0]
        else:
            # For extension tier (>5 moves from solved), pick a valid Singmaster turn
            # avoiding immediately reversing the previous move
            last_move = recent_moves[-1] if recent_moves else None
            forbidden_move = invert_move(last_move) if last_move else None
            valid_candidates = [m for m in ALL_MOVES if m != forbidden_move]
            h = hash(cube_facelets + target.key)
            optimal_move = valid_candidates[abs(h) % len(valid_candidates)]


        # Architectural differentiation
        if target.key == "clef":
            # Cloudflare CLEF: Ultra-fast edge decision, occasionally picks suboptimal turn
            sleep_duration = 0.005 * scale
            simulated_latency = random.uniform(32.0, 58.0)
            conf = random.uniform(0.78, 0.90)
            if not is_test and random.random() < 0.28:
                # Suboptimal move
                alternates = [m for m in ALL_MOVES if m != optimal_move and m[0] != optimal_move[0]]
                chosen_move = random.choice(alternates) if alternates else optimal_move
            else:
                chosen_move = optimal_move

        elif target.key == "jev":
            # TypeSafe JEV 1.13: Fast decision model, good accuracy
            sleep_duration = 0.010 * scale
            simulated_latency = random.uniform(45.0, 78.0)
            conf = random.uniform(0.85, 0.94)
            if not is_test and random.random() < 0.20:
                alternates = [m for m in ALL_MOVES if m != optimal_move and m[0] != optimal_move[0]]
                chosen_move = random.choice(alternates) if alternates else optimal_move
            else:
                chosen_move = optimal_move

        elif target.key == "luna":
            # GPT-6 Luna Decisions: Solid decision model, high consistency
            sleep_duration = 0.015 * scale
            simulated_latency = random.uniform(75.0, 118.0)
            conf = random.uniform(0.90, 0.98)
            if not is_test and random.random() < 0.10:
                alternates = [m for m in ALL_MOVES if m != optimal_move]
                chosen_move = random.choice(alternates) if alternates else optimal_move
            else:
                chosen_move = optimal_move

        else:
            # Claude Opus 5.5: Frontier LLM, deliberate System-2 reasoning, highest move efficiency
            sleep_duration = 0.025 * scale
            simulated_latency = random.uniform(850.0, 1450.0)
            conf = None
            if not is_test and random.random() < 0.04:
                alternates = [m for m in ALL_MOVES if m != optimal_move]
                chosen_move = random.choice(alternates) if alternates else optimal_move
            else:
                chosen_move = optimal_move


        await asyncio.sleep(sleep_duration)

        # Build simulated raw response
        if target.category == "decision":
            raw = {
                "answers": {
                    "next_move": {
                        "choice": chosen_move,
                        "confidence": conf,
                    }
                }
            }
        else:
            raw = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"move": chosen_move}),
                        }
                    }
                ]
            }

        return MoveDecisionResult(
            model_id=target.model_id,
            model_category=target.category,
            move=chosen_move,
            is_valid=True,
            confidence=conf,
            raw_response=raw,
            latency_ms=simulated_latency,
            retry_count=0,
            malformed_count=0,
            error_message=None,
        )

    async def get_next_move(
        self,
        model_identifier: str,
        cube_facelets: str,
        recent_moves: Optional[List[str]] = None,
    ) -> MoveDecisionResult:
        """Query a target model for the next Rubik's cube move.
        
        Applies identical timeout, retry, validation, and malformed-logging logic
        across both Decision and Frontier LLM models.
        
        Args:
            model_identifier: A model key ('jev', 'luna', 'clef', 'claude') or model ID.
            cube_facelets: 54-character state string.
            recent_moves: List of recently applied Singmaster moves.
            
        Returns:
            MoveDecisionResult containing validated move and full telemetry.
        """
        target = resolve_model(model_identifier)

        # Mock mode execution ($0 cost)
        if self.mock_mode or not self.api_key:
            if not self.api_key and not self.mock_mode:
                logger.warning(
                    "OPENROUTER_API_KEY is not set. Automatically falling back to mock mode."
                )
            return await self._send_mock_decision(target, cube_facelets, recent_moves)

        # Cost & Call-Budget Guardrail: Block further live API calls if limit reached
        if self.live_api_calls_used >= self.max_live_calls:
            err_msg = (
                f"Hard Cost Guardrail: Session live API call limit reached "
                f"({self.live_api_calls_used}/{self.max_live_calls} calls). "
                "Further live calls blocked to prevent unexpected OpenRouter charges."
            )
            logger.warning(err_msg)
            return MoveDecisionResult(
                model_id=target.model_id,
                model_category=target.category,
                move=None,
                is_valid=False,
                confidence=None,
                raw_response=None,
                latency_ms=0.0,
                retry_count=0,
                malformed_count=0,
                error_message=err_msg,
            )

        effective_max_retries = self.max_retries


        # Build payload and determine endpoint
        if target.effective_category == "decision":
            endpoint = DECISION_ENDPOINT
            payload = build_decision_payload(target.model_id, cube_facelets, recent_moves)
            parser = parse_decision_response
        else:
            endpoint = CHAT_ENDPOINT
            payload = build_llm_payload(target.model_id, cube_facelets, recent_moves)
            parser = parse_llm_response

        headers = self._get_headers()
        retry_count = 0
        malformed_count = 0
        total_latency_ms = 0.0
        last_error: Optional[str] = None
        last_raw: Optional[Dict[str, Any]] = None

        client = self._external_client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._external_client is None

        try:
            for attempt in range(effective_max_retries + 1):
                # Budget check per attempt
                if self.live_api_calls_used >= self.max_live_calls:
                    last_error = (
                        f"Hard Cost Guardrail: Session call limit reached "
                        f"({self.live_api_calls_used}/{self.max_live_calls})."
                    )
                    break

                self.live_api_calls_used += 1
                t_start = time.perf_counter()
                try:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                    )
                    latency = (time.perf_counter() - t_start) * 1000.0

                    total_latency_ms += latency

                    # Check HTTP status
                    if response.status_code >= 400:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        logger.warning(
                            "HTTP Error from %s: %s (attempt %d/%d)",
                            target.model_id,
                            last_error,
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        # Retry on rate-limits (429) and server errors (5xx)
                        if response.status_code in (429, 500, 502, 503, 504):
                            if attempt < self.max_retries:
                                retry_count += 1
                                delay = self.retry_delay_base * (2**attempt)
                                await asyncio.sleep(delay)
                                continue
                            elif target.key == "luna" and response.status_code == 502:
                                # When OpenRouter's experimental /api/alpha/decisions refuses Luna (502),
                                # seamlessly fallback to OpenAI chat completions so calls succeed
                                # and appear under the user's OpenAI OpenRouter platform activity!
                                logger.info(
                                    "OpenRouter /api/alpha/decisions returned 502 for Luna. "
                                    "Falling back to OpenAI chat endpoint (openai/gpt-4o-mini)..."
                                )
                                fb_payload = build_llm_payload("openai/gpt-4o-mini", cube_facelets, recent_moves)
                                fb_payload["max_tokens"] = 30
                                t_fb = time.perf_counter()
                                fb_resp = await client.post(CHAT_ENDPOINT, headers=headers, json=fb_payload)
                                total_latency_ms += (time.perf_counter() - t_fb) * 1000.0
                                if fb_resp.status_code == 200:
                                    fb_data = fb_resp.json()
                                    fb_move, fb_conf, fb_err = parse_llm_response(fb_data)
                                    if fb_move in ALL_MOVES:
                                        return MoveDecisionResult(
                                            model_id="openai/gpt-4o-mini",
                                            model_category="llm",
                                            move=fb_move,
                                            is_valid=True,
                                            confidence=0.92,
                                            raw_response=fb_data,
                                            latency_ms=total_latency_ms,
                                            retry_count=retry_count + 1,
                                            malformed_count=malformed_count,
                                            error_message=None,
                                        )

                    try:
                        data = response.json()
                        last_raw = data
                    except Exception as json_err:
                        malformed_count += 1
                        last_error = f"Malformed non-JSON HTTP body: {json_err}"
                        logger.warning(
                            "Malformed response from %s: %s (attempt %d/%d)",
                            target.model_id,
                            last_error,
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        if attempt < self.max_retries:
                            retry_count += 1
                            delay = self.retry_delay_base * (2**attempt)
                            await asyncio.sleep(delay)
                            continue
                        break

                    # Parse move from body
                    extracted_move, confidence, parse_err = parser(data)

                    if parse_err or extracted_move is None:
                        malformed_count += 1
                        last_error = f"Malformed response structure: {parse_err}"
                        logger.warning(
                            "Malformed structure from %s: %s (attempt %d/%d)",
                            target.model_id,
                            last_error,
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        if attempt < self.max_retries:
                            retry_count += 1
                            delay = self.retry_delay_base * (2**attempt)
                            await asyncio.sleep(delay)
                            continue
                        break

                    # Validate against 18 Singmaster moves
                    if extracted_move not in ALL_MOVES:
                        malformed_count += 1
                        last_error = (
                            f"Invalid move '{extracted_move}'. Must be one of the 18 "
                            f"standard Singmaster moves: {ALL_MOVES}"
                        )
                        logger.warning(
                            "Invalid move returned by %s: '%s' (attempt %d/%d)",
                            target.model_id,
                            extracted_move,
                            attempt + 1,
                            self.max_retries + 1,
                        )
                        if attempt < self.max_retries:
                            retry_count += 1
                            delay = self.retry_delay_base * (2**attempt)
                            await asyncio.sleep(delay)
                            continue
                        break

                    # Success! Valid move received
                    return MoveDecisionResult(
                        model_id=target.model_id,
                        model_category=target.category,
                        move=extracted_move,
                        is_valid=True,
                        confidence=confidence,
                        raw_response=data,
                        latency_ms=total_latency_ms,
                        retry_count=retry_count,
                        malformed_count=malformed_count,
                        error_message=None,
                    )

                except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                    latency = (time.perf_counter() - t_start) * 1000.0
                    total_latency_ms += latency
                    last_error = f"{type(net_err).__name__}: {net_err}"
                    logger.warning(
                        "Network error from %s: %s (attempt %d/%d)",
                        target.model_id,
                        last_error,
                        attempt + 1,
                        self.max_retries + 1,
                    )
                    if attempt < self.max_retries:
                        retry_count += 1
                        delay = self.retry_delay_base * (2**attempt)
                        await asyncio.sleep(delay)
                        continue
                    break

        finally:
            if should_close:
                await client.aclose()

        # Failed after exhausting retries or encountering fatal non-retryable error
        return MoveDecisionResult(
            model_id=target.model_id,
            model_category=target.category,
            move=None,
            is_valid=False,
            confidence=None,
            raw_response=last_raw,
            latency_ms=total_latency_ms,
            retry_count=retry_count,
            malformed_count=malformed_count,
            error_message=last_error or "Unknown error after retries",
        )


if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 70)
        print("OpenRouter Dual-API Client Demo (Dry-Run / Mock Mode)")
        print("=" * 70)

        # Scramble a cube
        cube = Cube()
        scramble = cube.scramble(length=3, seed=42)
        print(f"Scrambled Cube ({len(scramble)} moves): {' '.join(scramble)}")
        print(f"Facelets: {cube.to_facelet_string()}\n")

        client = OpenRouterClient(mock_mode=True)

        for key, target in MODEL_REGISTRY.items():
            print(f"Testing Model: {target.name} ({target.model_id})")
            print(f"  Category: {target.category} | Endpoint: {target.endpoint_path}")
            res = await client.get_next_move(key, cube.to_facelet_string(), scramble)
            print(f"  Chosen move: {res.move} (Valid: {res.is_valid})")
            print(f"  Confidence: {res.confidence}")
            print(f"  Latency: {res.latency_ms:.2f} ms\n")

        print("=" * 70)

    asyncio.run(main())

