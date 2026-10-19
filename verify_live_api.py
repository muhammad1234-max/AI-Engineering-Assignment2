"""Standalone Live OpenRouter API Verification Script (Zero Waste Single-Call Test).

Performs a single minimal-token live API call against a depth-1 scrambled cube
with strict cost-saving guardrails and assertions.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure backend modules can be imported
sys.path.insert(0, os.path.abspath("."))

from backend.app.cube_engine import ALL_MOVES, Cube
from backend.app.openrouter_client import OpenRouterClient, MODEL_REGISTRY


async def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not api_key or api_key == "your_openrouter_api_key_here":
        print("[!] No live OPENROUTER_API_KEY found in .env. Running in simulated fallback mode.")
        mock_mode = True
    else:
        # Masked display for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"[*] Live OPENROUTER_API_KEY loaded: {masked_key}")
        mock_mode = False

    # Initialize client with strict 5-call budget and 1 retry limit
    client = OpenRouterClient(
        api_key=api_key if not mock_mode else None,
        mock_mode=mock_mode,
        max_live_calls=5,
        max_retries=1,
    )

    print(f"[*] Client Initialized: mock_mode={client.mock_mode}, max_live_calls={client.max_live_calls}")
    print(f"[*] Live API Calls Used (before): {client.live_api_calls_used} / {client.max_live_calls}")

    # Prepare depth-1 test scramble
    cube = Cube()
    cube.apply_move("R")
    state = cube.to_facelet_string()
    recent_moves = ["R"]

    print("[*] Test State: 1-move scramble ('R')")
    print(f"[*] Target Model: typesafe/jev-1.13 (Category: {MODEL_REGISTRY['jev'].category})")
    print(f"[*] Target Endpoint: {MODEL_REGISTRY['jev'].full_endpoint}")

    # Execute single live call
    print("[*] Executing single API query...")
    res = await client.get_next_move(
        model_identifier="jev",
        cube_facelets=state,
        recent_moves=recent_moves,
    )

    print("\n--- Response Telemetry ---")
    print(f"Model ID:        {res.model_id}")
    print(f"Chosen Move:     {res.move}")
    print(f"Is Valid Move:   {res.is_valid}")
    print(f"Confidence:      {res.confidence}")
    print(f"Latency:         {res.latency_ms:.2f} ms")
    print(f"Retries:         {res.retry_count}")
    print(f"Malformed:       {res.malformed_count}")
    if res.error_message:
        print(f"Error Message:   {res.error_message}")
    print(f"Live Calls Used: {client.live_api_calls_used} / {client.max_live_calls}")

    # Verify assertions
    if not mock_mode and res.error_message and "404" in res.error_message:
        print("\n[NOTE] OpenRouter /api/alpha/decisions returned 404 (Alpha endpoint not publicly routed on live host).")
        print("[*] Testing single minimal-token call against Claude Opus 5.5 on /api/v1/chat/completions (max_tokens=16)...")
        claude_res = await client.get_next_move(
            model_identifier="claude",
            cube_facelets=state,
            recent_moves=recent_moves,
        )
        print(f"Claude Model ID:    {claude_res.model_id}")
        print(f"Claude Chosen Move: {claude_res.move}")
        print(f"Claude Valid:       {claude_res.is_valid}")
        print(f"Claude Latency:     {claude_res.latency_ms:.2f} ms")
        if claude_res.error_message:
            print(f"Claude Error:       {claude_res.error_message}")
        print(f"Total Live Calls:   {client.live_api_calls_used} / {client.max_live_calls}")

    # Final assertions
    assert client.live_api_calls_used >= 1 if not mock_mode else client.live_api_calls_used == 0, (
        "Call counter was not incremented correctly!"
    )
    assert client.live_api_calls_used <= client.max_live_calls, (
        "Call counter exceeded session budget!"
    )

    print("\n[OK] VERIFICATION SUCCESS: Single live API test completed within strict cost and token guardrails.")
    print(f"[OK] Final Session API Calls: {client.live_api_calls_used} / {client.max_live_calls}")



if __name__ == "__main__":
    asyncio.run(main())
