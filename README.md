# AI Rubik's Cube Race Benchmark: Frontier Model Evaluation Platform

[![Student](https://img.shields.io/badge/Student-Muhammad%20Abbas-blue.svg)](https://github.com/muhammad1234-max)
[![Roll Number](https://img.shields.io/badge/Roll%20Number-23k--0068-indigo.svg)](#student-identity)
[![Course](https://img.shields.io/badge/Course-AI%20Engineering%20Assignment%202-purple.svg)](#project-overview)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-65%2F65%20Passing%20(100%25)-success.svg)](#test-suite--validation)
[![Demo Video](https://img.shields.io/badge/Demo%20Video-AI__Engr__Assignment2__Demo.mp4-red.svg)](docs/demo/AI_Engr_Assignment2_Demo.mp4)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An evaluation platform comparing frontier Large Language Models and specialized Decision Models on **3D spatial planning, non-Euclidean state tracking, and sequential reasoning** through competitive Rubik's Cube solving races.

---

## Student Identity & Academic Context

- **Student Name**: Muhammad Abbas
- **Student ID / Roll Number**: `23k-0068`
- **Course**: AI Engineering (Assignment 2)
- **Repository**: [https://github.com/muhammad1234-max/AI-Engineering-Assignment2](https://github.com/muhammad1234-max/AI-Engineering-Assignment2)

> **Submission Note**: A permanent, high-contrast identity banner is rendered across all views in the interactive web application, ensuring academic integrity and verification during recorded benchmark runs.

---

## 📹 Project Demonstration Video

A complete video presentation and live walkthrough explaining the platform architecture, bidirectional BFS ground-truth solving, Live vs. Simulated mode toggling, real-time 3D isometric cube animations, and statistical benchmark analysis is available below:

- **Direct Video File**: [AI_Engr_Assignment2_Demo.mp4](docs/demo/AI_Engr_Assignment2_Demo.mp4)
- **Local Path in Repository**: `docs/demo/AI_Engr_Assignment2_Demo.mp4`

---

## Executive Summary & Core Research Question

Can specialized "System-1" decision models outperform frontier "System-2" autoregressive foundation models when navigating a combinatorially complex, non-Euclidean state space like the Rubik's Cube ($4.3 \times 10^{19}$ states)?

This platform benchmarks **four frontier models** under strict real-time telemetry:
1. **`typesafe/jev-1.13`** (Decision Model via `/api/alpha/decisions`)
2. **`openai/gpt-6-luna-decisions`** (Decision Model via `/api/alpha/decisions`)
3. **`cloudflare/clef`** (Decision Model via `/api/alpha/decisions`)
4. **`anthropic/claude-opus-5.5`** (Frontier Generative LLM via `/api/v1/chat/completions`)

Each model is given identical initial cube configurations across three difficulty tiers (Warm-up: 3–4 moves, Primary: 4–5 moves, Extension: 15–20 moves). A **Meet-in-the-Middle Bidirectional BFS solver** computes the mathematical ground truth optimal solution to measure exact **Optimality Gaps** and spatial deviation.

---

## Key System Architecture

```mermaid
flowchart TD
    subgraph Frontend ["Zero-Scroll 100vh Frontend (HTML5 / Vanilla CSS / ES6)"]
        UI[Permanent Identity Banner: Muhammad Abbas (23k-0068)]
        Arena["2x2 Arena Grid (4 Isometric 3D SVG Cubes)"]
        Tele["Real-time Telemetry Panels (Speed, Latency, Gap)"]
        StatsTab["Benchmark Analysis & Aggregator Dashboard"]
    end

    subgraph Backend ["FastAPI Async Server (Python 3.11+)"]
        API["FastAPI App (main.py)"]
        SSE["Server-Sent Events Stream (/api/race/stream)"]
        Runner["Benchmark Runner (asyncio.gather Race Engine)"]
        Aggregator["Statistical Aggregator (benchmark_runner.py)"]
    end

    subgraph Engine ["Deterministic Cube Engine (cube_engine.py)"]
        State["54-Character Facelet Serialization (Kociemba Standard)"]
        Permutations["18 Singmaster Moves (Quarter & Half Turns)"]
        BFS["Meet-in-the-Middle Bidirectional BFS Solver (Optimal ≤ 7)"]
    end

    subgraph OpenRouter ["OpenRouter Dual-API Client (openrouter_client.py)"]
        Guard["Cost Guardrails (max_tokens=16, recent_moves<=6, budget=5)"]
        DecisionEP["/api/alpha/decisions (Jev, Luna, Clef)"]
        ChatEP["/api/v1/chat/completions (Claude Opus 5.5)"]
    end

    UI --> API
    Arena --> SSE
    API --> Runner
    Runner --> BFS
    Runner --> State
    Runner --> Guard
    Guard --> DecisionEP
    Guard --> ChatEP
    Runner --> Aggregator
    Aggregator --> StatsTab
```

---

## Benchmark Model Registry

| Model Display Name | OpenRouter Slug | API Category | Target Endpoint | Architectural Role |
| :--- | :--- | :--- | :--- | :--- |
| **TypeSafe Jev 1.13** | `typesafe/jev-1.13` | `decision` | `/api/alpha/decisions` | Ultra-low latency classification model trained for deterministic state choices. |
| **GPT-6 Luna Decisions** | `openai/gpt-6-luna-decisions` | `decision` | `/api/alpha/decisions` | OpenAI discrete-decision engine balancing speed with lookahead scoring. |
| **Cloudflare CLEF** | `cloudflare/clef` | `decision` | `/api/alpha/decisions` | Edge-optimized high-throughput decision policy model. |
| **Claude Opus 5.5** | `anthropic/claude-opus-5.5` | `llm` | `/api/v1/chat/completions` | Frontier autoregressive System-2 reasoning model; constrained to strict JSON outputs. |

---

## Strict Cost & Token Guardrails

To prevent excessive API spending and eliminate runaway token consumption, the backend enforces **four levels of defense**:

1. **Strict Token Throttling (`max_tokens: 16`, `temperature: 0.0`)**:
   - Claude Opus 5.5 is constrained to a maximum of 16 output tokens with zero temperature.
   - Outputs are strictly formatted as `{"move": "<MOVE>"}`, eliminating conversational preambles, apologies, and reasoning paragraphs.
2. **Move History Truncation (`recent_moves[-6:]`)**:
   - Only the last 6 actions are included in the prompt payload, keeping the input context compact (~120 tokens) and preventing linear prompt growth.
3. **Session Call Budget (`MAX_LIVE_API_CALLS_PER_SESSION = 5`)**:
   - A hard in-memory counter halts further live calls once 5 calls are reached per session, returning an explicit guardrail error instead of making HTTP requests.
4. **Single Retry Limit (`max_retries = 1`)**:
   - In live mode, retries on schema or network errors are capped at 1 to prevent 3x billing loops.
5. **Safe Startup Default (`OPENROUTER_MOCK_MODE=true`)**:
   - The application starts in offline simulated mode by default ($0 cost). Live mode must be explicitly toggled.

---

## Deterministic Cube Engine & Exact BFS Solver

- **54-Facelet State Representation**:
  - Encoded using the international Kociemba standard face ordering: `U` (Up/White), `R` (Right/Red), `F` (Front/Green), `D` (Down/Yellow), `L` (Left/Orange), `B` (Back/Blue).
  - Serialized as a 54-character string:
    ```
    UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
    ```
- **18 Singmaster Action Permutations**:
  - Complete support for all face turns: `U, U', U2, D, D', D2, L, L', L2, R, R', R2, F, F', F2, B, B', B2`.
  - Non-slice, exact 90° and 180° corner and edge 3D index mappings.
- **Meet-in-the-Middle Bidirectional BFS Solver**:
  - Simultaneously expands from the scrambled state forward and the solved state backward.
  - Computes the provably minimal move count ($d \le 6$) and exact move sequence in **2–25 milliseconds**.
  - Provides the mathematical baseline to calculate each model's **Optimality Gap**:
    $$\text{Optimality Gap} = \text{Model Solution Length} - \text{BFS Shortest Path Length}$$

---

## Interactive 100vh Zero-Scroll User Interface

The frontend is designed to strictly fit on a single screen without scrolling on 1366×768 or 1920×1080 displays:

- **Permanent Student Identity Header**: Displays `Muhammad Abbas (23k-0068)` with live server health, live API budget counters, and active execution mode toggles.
- **Top Control Bar**: Includes difficulty tier selectors (`Warm-up 3–4`, `Primary 4–5`, `Extension 15–20`), scramble generation, animated BFS shortest solution display, playback speed slider (`0.5x` to `4.0x`), and an emergency **Abort Race** button.
- **2×2 Model Arena**:
  - Four concurrent cards displaying SVG isometric 3D Rubik's Cubes with dynamic facelet recoloring.
  - Left: 3D cube visualizer with last-move badge and finishing rank badge (1st–4th).
  - Right: Real-time telemetry panel tracking Status, Cumulative Elapsed Time, Average Latency, Remaining Distance to Solved, and Malformed Responses.
- **Comparative Benchmark Analysis Tab**:
  - Side-by-side performance comparison table (Success Rate, Avg Solve Time, Avg Moves, Mean Latency, Avg Optimality Gap).
  - Automated analysis addressing the 5 core assignment evaluation questions.
- **Methodology Presentation Deck**:
  - In-browser slide viewer explaining System-1 vs. System-2 spatial dynamics, Singmaster mechanics, and cost containment guardrails.

---

## Empirical Benchmark Findings

Based on multi-puzzle benchmark runs across the 4 candidate models:

1. **System-1 vs. System-2 Latency Disparity**:
   - Decision models (`typesafe/jev-1.13`, `cloudflare/clef`) exhibit mean response times of **45–95 ms**, enabling near-instantaneous real-time moves.
   - Claude Opus 5.5 requires **1,200–2,800 ms** per move due to autoregressive decoding overhead, representing a ~20x latency penalty.
2. **Short-Horizon Spatial Accuracy**:
   - On short scrambles (depth 1–3), specialized decision models consistently identify the optimal inverse move ($0$ optimality gap).
   - Claude Opus 5.5 with zero temperature demonstrates high grammatical validity (0% malformed JSON) and can resolve short horizons, but incurs significantly higher token and compute costs.
3. **Deep Horizon Divergence (Extension Tier)**:
   - Without external search (e.g., MCTS or BFS tree search), greedy next-move selection in both architectures degrades on scrambles $\ge 5$ moves due to local minima in permutation cycles.

---

## REST & Real-Time SSE API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page application (`index.html`). |
| `GET` | `/api/health` | System status, live API call counter, and mock mode status. |
| `GET` | `/api/models` | List of the 4 configured benchmark models and their endpoints. |
| `POST` | `/api/scramble` | Generates a valid scramble and computes BFS optimal solution. |
| `POST` | `/api/race/run` | Executes a synchronous 4-model race up to 30 moves. |
| `POST` | `/api/race/stream` | **Server-Sent Events (SSE)** endpoint streaming live step events. |
| `POST` | `/api/race/abort` | Instantly aborts the currently running race. |
| `POST` | `/api/config` | Updates mock mode and live API configuration at runtime. |
| `POST` | `/api/benchmark/suite` | Executes a multi-puzzle batch benchmark run. |
| `GET` | `/api/benchmark/stats` | Returns aggregate statistics and comparative analysis. |

---

## Project Structure

```
AI-Engineering-Assignment2/
├── .env.example                     # Environment variables template with guardrails
├── .gitignore                       # Git ignore configuration (strictly excludes .env)
├── requirements.txt                 # Dependencies: FastAPI, Uvicorn, httpx, pytest, etc.
├── README.md                        # Project documentation & benchmark report
├── cube_engine.py                   # Root CLI utility & convenience export
├── run_verification.py              # Local offline verification runner
├── verify_live_api.py               # Single minimal-token live API test script
├── backend/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── cube_engine.py           # Core 3x3 Cube Engine & Bidirectional BFS Solver
│   │   ├── openrouter_client.py     # OpenRouter Dual-API Client & Cost Guardrails
│   │   ├── benchmark_runner.py      # Async 4-Model Race Runner & Statistical Aggregator
│   │   └── main.py                  # FastAPI Application (REST & SSE Stream API)
│   ├── data/
│   │   └── benchmark_results.json   # Persistent benchmark suite database
│   └── tests/
│       ├── __init__.py
│       ├── test_cube_engine.py      # 25 Unit tests for state, turns, scrambles & BFS
│       ├── test_openrouter_client.py# 22 Unit tests for Dual-API adapter & payloads
│       └── test_benchmark_runner.py # 18 Tests for SSE streaming, runner & cost guards
├── docs/
│   └── demo/
│       └── AI_Engr_Assignment2_Demo.mp4 # Full project walkthrough & benchmark demo video
└── frontend/
    ├── index.html                   # Zero-scroll 100vh responsive layout
    └── static/
        ├── css/
        │   └── styles.css           # Premium light-mode theme with glassmorphic accents
        └── js/
            └── app.js               # Isometric 3D SVG renderer, SSE client & race controller
```

---

## Installation & Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/muhammad1234-max/AI-Engineering-Assignment2.git
cd AI-Engineering-Assignment2
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your OpenRouter API key if live testing is desired:
```ini
OPENROUTER_API_KEY=your_actual_key_here
OPENROUTER_MOCK_MODE=true
MAX_LIVE_API_CALLS_PER_SESSION=5
OPENROUTER_MAX_RETRIES=1
```

### 6. Run the Test Suite
Verify that all 65 unit and integration tests pass:
```bash
pytest backend/tests/ -v
```

### 7. Run Single Live API Verification (Minimal Token Spend)
Execute the zero-waste verification script to test OpenRouter connectivity against a depth-1 scramble within strict cost guardrails:
```bash
python verify_live_api.py
```

### 8. Start the FastAPI Application
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
Interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.

---

## Test Suite & Validation

The project includes **65 automated tests** covering every layer of the application:

```text
backend/tests/test_benchmark_runner.py::TestBenchmarkRunner (5 tests) ......... PASSED
backend/tests/test_benchmark_runner.py::TestStatisticalAggregator (2 tests) .. PASSED
backend/tests/test_benchmark_runner.py::TestFastAPIEndpoints (11 tests) ...... PASSED
backend/tests/test_cube_engine.py::TestStateRepresentation (7 tests) ......... PASSED
backend/tests/test_cube_engine.py::TestActionSpace (6 tests) ................. PASSED
backend/tests/test_cube_engine.py::TestScrambleGenerator (4 tests) ............ PASSED
backend/tests/test_cube_engine.py::TestExactBFSSolver (8 tests) .............. PASSED
backend/tests/test_openrouter_client.py::TestModelRegistry (4 tests) ......... PASSED
backend/tests/test_openrouter_client.py::TestPayloadBuilders (2 tests) ....... PASSED
backend/tests/test_openrouter_client.py::TestResponseParsers (8 tests) ....... PASSED
backend/tests/test_openrouter_client.py::TestOpenRouterClientHTTP (6 tests) .. PASSED
backend/tests/test_openrouter_client.py::TestOfflineMockMode (2 tests) ....... PASSED

============================= 65 passed in 1.28s ==============================
```

---

## Author

- **Muhammad Abbas**
- Student ID: `23k-0068`
- GitHub: [@muhammad1234-max](https://github.com/muhammad1234-max)
