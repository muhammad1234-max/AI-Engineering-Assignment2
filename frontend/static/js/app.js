/**
 * Main Application Logic for AI Rubik's Cube Race Benchmark.
 * Matches Page 4 Light Theme and Zero-Scroll 100vh Arena specification.
 */

const MODELS = {
  claude: {
    key: 'claude',
    name: 'Claude Opus 5.5',
    category: 'LLM',
    badgeClass: 'badge-llm',
    badgeText: 'LLM',
    thinkingText: 'thinking..',
  },
  jev: {
    key: 'jev',
    name: 'TypeSafe JEV 1.13',
    category: 'DECISION MODEL',
    badgeClass: 'badge-decision',
    badgeText: 'DECISION MODEL',
    thinkingText: 'deciding..',
  },
  luna: {
    key: 'luna',
    name: 'GPT-6 Luna Decisions',
    category: 'DECISION MODEL',
    badgeClass: 'badge-decision',
    badgeText: 'DECISION MODEL',
    thinkingText: 'deciding..',
  },
  clef: {
    key: 'clef',
    name: 'Cloudflare CLEF',
    category: 'DECISION MODEL',
    badgeClass: 'badge-decision',
    badgeText: 'DECISION MODEL',
    thinkingText: 'deciding..',
  },
};

// Application State
const state = {
  currentScramble: ['B', 'U2', 'D', 'F'],
  initialFacelets: 'UUUUUUBLDRLLRRDRFFUFBLFBLFBFRULDDLDDRRLULDBBDFFFBBBRRD',
  shortestDistance: 4,
  shortestMoves: ["F'", "D'", "U2", "B'"],
  activeTab: 'arena',
  playbackSpeed: 1.0,
  isRacing: false,
  isMockMode: true,
  liveApiCallsUsed: 0,
  maxLiveCalls: 160,
  abortController: null,
  models: {},
};


// Initialize model state objects
Object.keys(MODELS).forEach((key) => {
  state.models[key] = {
    facelets: state.initialFacelets,
    moves: [],
    moveCount: 0,
    startTime: 0,
    elapsedMs: 0,
    timerInterval: null,
    status: 'READY',
    rank: null,
    optimalityGap: null,
    latencyMs: 0,
    isSolved: false,
  };
});

// Format milliseconds as M:SS.CC (matching Page 4 screenshot: 0:14.75)
function formatStopwatch(ms) {
  const totalSecs = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSecs / 60);
  const seconds = totalSecs % 60;
  const centiseconds = Math.floor((ms % 1000) / 10);
  return `${minutes}:${seconds.toString().padStart(2, '0')}.${centiseconds.toString().padStart(2, '0')}`;
}

// ==========================================================================
// Initialization & DOM Setup
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupControls();
  setupConfigModal();
  renderAllCubes();
  updateScrambleDisplay();
  checkBackendHealth();
  loadInitialBenchmarkStats();
});

function setupNavigation() {
  const tabArena = document.getElementById('tab-arena');
  const tabBenchmark = document.getElementById('tab-benchmark');
  const viewArena = document.getElementById('view-arena');
  const viewBenchmark = document.getElementById('view-benchmark');

  tabArena.addEventListener('click', () => {
    state.activeTab = 'arena';
    tabArena.classList.add('active');
    tabBenchmark.classList.remove('active');
    viewArena.style.display = 'grid';
    viewBenchmark.style.display = 'none';
  });

  tabBenchmark.addEventListener('click', () => {
    state.activeTab = 'benchmark';
    tabBenchmark.classList.add('active');
    tabArena.classList.remove('active');
    viewArena.style.display = 'none';
    viewBenchmark.style.display = 'block';
    loadInitialBenchmarkStats();
  });
}

function setupControls() {
  const btnGenerate = document.getElementById('btn-generate-scramble');
  const btnStartRace = document.getElementById('btn-start-race');
  const tierSelect = document.getElementById('tier-select');
  const btnRunSuite = document.getElementById('btn-run-suite');

  btnGenerate.addEventListener('click', async () => {
    if (state.isRacing) return;
    const tier = tierSelect.value;
    btnGenerate.disabled = true;
    try {
      const res = await fetch('/api/scramble', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tier }),
      });
      const data = await res.json();
      applyNewScramble(data);
    } catch (err) {
      console.error('Failed to generate scramble:', err);
    } finally {
      btnGenerate.disabled = false;
    }
  });

  btnStartRace.addEventListener('click', () => {
    if (state.isRacing) return;
    startLiveRace();
  });

  const btnAbort = document.getElementById('btn-abort-race');
  if (btnAbort) {
    btnAbort.addEventListener('click', async () => {
      if (!state.isRacing) return;
      await abortLiveRace();
    });
  }

  if (btnRunSuite) {
    btnRunSuite.addEventListener('click', () => {
      runBenchmarkSuiteSuite(5);
    });
  }

  // Playback speed buttons
  const speedButtons = document.querySelectorAll('.speed-btn');
  speedButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      speedButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.playbackSpeed = parseFloat(btn.dataset.speed || '1.0');
    });
  });
}

async function abortLiveRace() {
  if (state.abortController) {
    try {
      state.abortController.abort();
    } catch (e) {}
  }
  try {
    await fetch('/api/race/abort', { method: 'POST' });
  } catch (err) {
    console.warn('Failed to call race abort endpoint:', err);
  }
  finishRaceAborted();
}

function finishRaceAborted() {
  state.isRacing = false;
  const btnStart = document.getElementById('btn-start-race');
  const btnGen = document.getElementById('btn-generate-scramble');
  const btnAbort = document.getElementById('btn-abort-race');

  if (btnStart) btnStart.disabled = false;
  if (btnGen) btnGen.disabled = false;
  if (btnAbort) btnAbort.disabled = true;

  Object.keys(MODELS).forEach((key) => {
    const m = state.models[key];
    if (m.timerInterval) clearInterval(m.timerInterval);
    if (m.status === 'THINKING' || m.status === 'IN_PROGRESS') {
      m.status = 'ABORTED';
    }
    updateCardUI(key);
  });
  checkBackendHealth();
}


// ==========================================================================
// Mode Switcher & API Key Configuration
// ==========================================================================

function setupConfigModal() {
  const btnMode = document.getElementById('btn-mode-modal');
  const modal = document.getElementById('config-modal');
  const btnClose = document.getElementById('btn-close-modal');
  const btnCancel = document.getElementById('btn-cancel-modal');
  const btnSave = document.getElementById('btn-save-modal');
  const selectMode = document.getElementById('modal-mode-select');
  const inputApiKey = document.getElementById('modal-api-key');

  btnMode.addEventListener('click', () => {
    selectMode.value = state.isMockMode ? 'mock' : 'live';
    modal.classList.add('open');
  });

  const closeModal = () => modal.classList.remove('open');
  btnClose.addEventListener('click', closeModal);
  btnCancel.addEventListener('click', closeModal);

  btnSave.addEventListener('click', async () => {
    const isMock = selectMode.value === 'mock';
    const keyVal = inputApiKey.value.trim();

    try {
      const payload = { mock_mode: isMock };
      if (keyVal) payload.api_key = keyVal;

      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      state.isMockMode = data.mock_mode;
      if (data.live_api_calls_used !== undefined) {
        state.liveApiCallsUsed = data.live_api_calls_used;
        state.maxLiveCalls = data.max_live_calls;
        updateBudgetBadgeUI(data.live_api_calls_used, data.max_live_calls);
      }
      updateModeButtonUI();
      closeModal();
    } catch (err) {
      console.error('Failed to update config:', err);
    }
  });
}

async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    state.isMockMode = data.mock_mode;
    if (data.live_api_calls_used !== undefined) {
      state.liveApiCallsUsed = data.live_api_calls_used;
      state.maxLiveCalls = data.max_live_calls;
      updateBudgetBadgeUI(data.live_api_calls_used, data.max_live_calls);
    }
    updateModeButtonUI();
  } catch (err) {
    console.warn('Backend health check error:', err);
  }
}

function updateBudgetBadgeUI(used, max) {
  const valEl = document.getElementById('val-api-calls');
  const maxEl = document.getElementById('val-max-calls');
  const pill = document.getElementById('pill-api-budget');
  const curUsed = used ?? state.liveApiCallsUsed;
  const curMax = max ?? state.maxLiveCalls;

  if (valEl) valEl.textContent = curUsed;
  if (maxEl) maxEl.textContent = curMax;

  if (pill) {
    if (curUsed >= curMax && curMax > 0) {
      pill.style.borderColor = '#EF4444';
      pill.style.background = '#450A0A';
    } else {
      pill.style.borderColor = 'rgba(255, 255, 255, 0.1)';
      pill.style.background = '#1E293B';
    }
  }
}

function updateModeButtonUI() {
  const btnMode = document.getElementById('btn-mode-modal');
  if (btnMode) {
    if (state.isMockMode) {
      btnMode.textContent = 'Mode: Simulated';
      btnMode.style.borderColor = '#94A3B8';
      btnMode.style.color = '#F8FAFC';
    } else {
      btnMode.textContent = 'Mode: Live OpenRouter';
      btnMode.style.borderColor = '#10B981';
      btnMode.style.color = '#34D399';
    }
  }
}


// ==========================================================================
// Scramble Application & UI Sync
// ==========================================================================

function applyNewScramble(data) {
  state.currentScramble = data.scramble;
  state.initialFacelets = data.initial_state;
  state.shortestDistance = data.shortest_distance;
  state.shortestMoves = data.shortest_moves;

  updateScrambleDisplay();

  // Reset all 4 model states
  Object.keys(MODELS).forEach((key) => {
    const m = state.models[key];
    if (m.timerInterval) clearInterval(m.timerInterval);
    m.facelets = state.initialFacelets;
    m.moves = [];
    m.moveCount = 0;
    m.elapsedMs = 0;
    m.status = 'READY';
    m.rank = null;
    m.optimalityGap = null;
    m.latencyMs = 0;
    m.isSolved = false;
    updateCardUI(key);
  });
}

function updateScrambleDisplay() {
  const scrambleChipsContainer = document.getElementById('scramble-chips');
  const groundTruthChipsContainer = document.getElementById('ground-truth-chips');
  const bfsDistanceBadge = document.getElementById('bfs-distance-badge');

  if (scrambleChipsContainer) {
    scrambleChipsContainer.innerHTML = state.currentScramble
      .map((m) => `<span class="move-chip">${m}</span>`)
      .join('');
  }

  if (groundTruthChipsContainer) {
    if (state.shortestMoves && state.shortestMoves.length > 0) {
      groundTruthChipsContainer.innerHTML = state.shortestMoves
        .map((m) => `<span class="ground-truth-chip">${m}</span>`)
        .join('');
    } else {
      groundTruthChipsContainer.innerHTML = `<span class="ground-truth-chip">${state.shortestDistance} moves</span>`;
    }
  }

  if (bfsDistanceBadge) {
    bfsDistanceBadge.textContent = `${state.shortestDistance} Moves`;
  }
}

// ==========================================================================
// 3D Isometric Cube Rendering
// ==========================================================================

function renderAllCubes() {
  Object.keys(MODELS).forEach((key) => {
    renderModelCube(key);
  });
}

function renderModelCube(key) {
  const m = state.models[key];
  const container = document.getElementById(`cube-${key}`);
  if (container) {
    container.innerHTML = renderIsometricCubeSVG(m.facelets);
  }
}

// ==========================================================================
// Card UI Updates (Matching Page 4 Light Theme)
// ==========================================================================

function updateCardUI(key) {
  const m = state.models[key];
  const card = document.getElementById(`card-${key}`);
  if (!card) return;

  // 1. Status Indicator Text
  const statusEl = document.getElementById(`status-${key}`);
  if (statusEl) {
    statusEl.className = 'status-badge';
    if (m.status === 'READY') {
      statusEl.textContent = 'READY';
    } else if (m.status === 'THINKING' || m.status === 'IN_PROGRESS') {
      statusEl.classList.add('thinking');
      statusEl.textContent = MODELS[key].thinkingText;
    } else if (m.status === 'SOLVED') {
      statusEl.textContent = 'SOLVED ✓';
    } else if (m.status === 'ABORTED') {
      statusEl.textContent = 'ABORTED';
      statusEl.style.color = 'var(--color-danger)';
    } else if (m.status === 'RAN_OUT_OF_MOVES' || m.status === 'ERROR') {
      statusEl.textContent = 'NOT SOLVED';
    }

  }

  // 2. Card Border & Telemetry Box Tints (Matching Page 4 Reference States)
  card.classList.remove('status-solved', 'status-failed');
  if (m.status === 'SOLVED') {
    card.classList.add('status-solved');
  } else if (m.status === 'RAN_OUT_OF_MOVES' || m.status === 'ERROR') {
    card.classList.add('status-failed');
  }

  // 3. Telemetry values (TIME, MOVES, Latency)
  const timeEl = document.getElementById(`time-${key}`);
  const movesEl = document.getElementById(`moves-${key}`);
  const latencyEl = document.getElementById(`latency-${key}`);

  if (timeEl) timeEl.textContent = formatStopwatch(m.elapsedMs);
  if (movesEl) movesEl.textContent = `${m.moveCount} / 30`;
  if (latencyEl) {
    latencyEl.textContent = m.latencyMs > 0 ? `${m.latencyMs.toFixed(0)}ms` : '--';
  }

  // 4. Rank Medallion Badge
  const rankOverlay = document.getElementById(`rank-${key}`);
  if (rankOverlay) {
    rankOverlay.className = 'rank-badge-overlay';
    if (m.rank) {
      rankOverlay.classList.add('visible');
      const rankSuffix = ['TH', 'ST', 'ND', 'RD', 'TH'][m.rank] || 'TH';
      rankOverlay.classList.add(`rank-badge-${m.rank}${rankSuffix.toLowerCase()}`);
      rankOverlay.textContent = `${m.rank}${rankSuffix}`;
    }
  }

  // 5. Update Cube SVG
  renderModelCube(key);
}

// ==========================================================================
// Live SSE Race Execution
// ==========================================================================

async function startLiveRace() {
  state.isRacing = true;
  document.getElementById('btn-start-race').disabled = true;
  document.getElementById('btn-generate-scramble').disabled = true;
  const btnAbort = document.getElementById('btn-abort-race');
  if (btnAbort) btnAbort.disabled = false;

  state.abortController = new AbortController();

  // Reset and start individual model timers
  Object.keys(MODELS).forEach((key) => {
    const m = state.models[key];
    if (m.timerInterval) clearInterval(m.timerInterval);
    m.moves = [];
    m.moveCount = 0;
    m.elapsedMs = 0;
    m.status = 'THINKING';
    m.rank = null;
    m.optimalityGap = null;
    m.isSolved = false;
    updateCardUI(key);

    m.startTime = Date.now();
    m.timerInterval = setInterval(() => {
      if (m.status === 'THINKING' || m.status === 'IN_PROGRESS') {
        m.elapsedMs = Date.now() - m.startTime;
        const timeEl = document.getElementById(`time-${key}`);
        if (timeEl) timeEl.textContent = formatStopwatch(m.elapsedMs);
      }
    }, 40);
  });

  try {
    const response = await fetch('/api/race/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scramble: state.currentScramble,
        max_moves: 30,
      }),
      signal: state.abortController.signal,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const evText of events) {
        if (!evText.trim()) continue;
        parseAndApplySSE(evText);
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('SSE connection error:', err);
    }
  } finally {
    state.isRacing = false;
    document.getElementById('btn-start-race').disabled = false;
    document.getElementById('btn-generate-scramble').disabled = false;
    if (btnAbort) btnAbort.disabled = true;
    state.abortController = null;

    // Ensure all timers stop
    Object.keys(MODELS).forEach((key) => {
      const m = state.models[key];
      if (m.timerInterval) clearInterval(m.timerInterval);
    });
    checkBackendHealth();
  }
}

function parseAndApplySSE(eventChunk) {
  const lines = eventChunk.split('\n');
  let eventType = 'message';
  let dataStr = '';

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataStr += line.slice(5).trim();
    }
  }

  if (!dataStr) return;

  try {
    const payload = JSON.parse(dataStr);
    handleRaceEvent(eventType, payload);
  } catch (err) {
    console.warn('Failed to parse SSE payload:', err);
  }
}

function handleRaceEvent(eventType, payload) {
  if (eventType === 'race_aborted') {
    finishRaceAborted();
    return;
  }

  if (eventType === 'move_step') {
    const { model_key, step, move, cube_facelets, is_solved, latency_ms, status } = payload;
    const m = state.models[model_key];
    if (!m) return;

    if (move) m.moves.push(move);
    m.moveCount = step;
    m.facelets = cube_facelets;
    m.latencyMs = latency_ms;
    m.status = status;

    if (is_solved) {
      m.isSolved = true;
      if (m.timerInterval) clearInterval(m.timerInterval);
    } else if (status === 'RAN_OUT_OF_MOVES' || status === 'ERROR' || status === 'ABORTED') {
      if (m.timerInterval) clearInterval(m.timerInterval);
    }

    updateCardUI(model_key);
    checkBackendHealth();
  } else if (eventType === 'race_complete') {

    const { participants } = payload;
    if (participants) {
      Object.entries(participants).forEach(([key, p]) => {
        const m = state.models[key];
        if (!m) return;
        m.rank = p.rank;
        m.status = p.status;
        m.optimalityGap = p.optimality_gap;
        m.elapsedMs = p.total_time_ms;
        if (m.timerInterval) clearInterval(m.timerInterval);
        updateCardUI(key);
      });
    }
  }
}

// ==========================================================================
// Benchmark Suite & Presentation Slide View
// ==========================================================================

async function loadInitialBenchmarkStats() {
  try {
    const res = await fetch('/api/benchmark/stats');
    const data = await res.json();
    renderBenchmarkDashboard(data);
  } catch (err) {
    console.error('Failed to load benchmark stats:', err);
  }
}

async function runBenchmarkSuiteSuite(numPuzzles = 5) {
  const btnRun = document.getElementById('btn-run-suite');
  const spinner = document.getElementById('suite-spinner');
  if (btnRun) btnRun.disabled = true;
  if (spinner) spinner.style.display = 'inline-block';

  try {
    const res = await fetch('/api/benchmark/suite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_puzzles: numPuzzles, tier: 'primary' }),
    });
    const data = await res.json();
    renderBenchmarkDashboard(data);
  } catch (err) {
    console.error('Failed to execute benchmark suite:', err);
  } finally {
    if (btnRun) btnRun.disabled = false;
    if (spinner) spinner.style.display = 'none';
  }
}

function renderBenchmarkDashboard(data) {
  const { model_stats, analysis_insights } = data;
  if (!model_stats) return;

  // 1. Populate Section 3C Summary Table
  const tbody = document.getElementById('benchmark-table-body');
  if (tbody) {
    tbody.innerHTML = Object.values(model_stats)
      .map((stats) => {
        const gapText =
          stats.avg_optimality_gap !== null
            ? stats.avg_optimality_gap === 0
              ? '0.0 (Optimal)'
              : `+${stats.avg_optimality_gap.toFixed(1)}`
            : 'N/A';

        const badgeClass = stats.category === 'llm' ? 'badge-llm' : 'badge-decision';

        return `
          <tr>
            <td>
              <div class="model-cell">
                <span class="name">${stats.model_name}</span>
                <span class="id">${stats.model_id}</span>
              </div>
            </td>
            <td>
              <span class="category-badge ${badgeClass}">${stats.category.toUpperCase()}</span>
            </td>
            <td><strong>${stats.avg_latency_ms.toFixed(1)} ms</strong></td>
            <td>${stats.avg_solve_time_seconds.toFixed(3)} s</td>
            <td>${stats.avg_moves.toFixed(1)}</td>
            <td><strong style="color: #16A34A">${stats.success_rate_pct.toFixed(0)}%</strong></td>
            <td><strong>${gapText}</strong></td>
            <td>${stats.total_malformed} / ${stats.total_retries}</td>
            <td>
              <div class="rank-pills">
                <span class="rank-mini-pill" style="background:#FEF3C7;color:#92400E">${stats.first_place_finishes}×1st</span>
                <span class="rank-mini-pill" style="background:#F1F5F9;color:#334155">${stats.second_place_finishes}×2nd</span>
                <span class="rank-mini-pill" style="background:#FFEDD5;color:#9A3412">${stats.third_place_finishes}×3rd</span>
              </div>
            </td>
          </tr>
        `;
      })
      .join('');
  }

  // 2. Populate Comparison Bar Meters
  const speedMeters = document.getElementById('speed-comparison-meters');
  const efficiencyMeters = document.getElementById('efficiency-comparison-meters');

  if (speedMeters) {
    const maxLatency = Math.max(...Object.values(model_stats).map((s) => s.avg_latency_ms), 100);
    speedMeters.innerHTML = Object.values(model_stats)
      .map((s) => {
        const pct = Math.min(100, Math.max(5, (s.avg_latency_ms / maxLatency) * 100));
        const color = s.category === 'llm' ? '#6366F1' : '#0EA5E9';
        return `
          <div class="bar-item">
            <div class="bar-label-row">
              <span>${s.model_name}</span>
              <span style="font-family: var(--font-mono)">${s.avg_latency_ms.toFixed(0)} ms / move</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width: ${pct}%; background: ${color}"></div>
            </div>
          </div>
        `;
      })
      .join('');
  }

  if (efficiencyMeters) {
    efficiencyMeters.innerHTML = Object.values(model_stats)
      .map((s) => {
        const gapVal = s.avg_optimality_gap !== null ? s.avg_optimality_gap : 10;
        const pct = Math.min(100, Math.max(10, (1 - gapVal / 10) * 100));
        return `
          <div class="bar-item">
            <div class="bar-label-row">
              <span>${s.model_name}</span>
              <span style="font-family: var(--font-mono)">Gap: ${s.avg_optimality_gap !== null ? s.avg_optimality_gap.toFixed(1) : 'Failed'} moves</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width: ${pct}%; background: #16A34A"></div>
            </div>
          </div>
        `;
      })
      .join('');
  }

  // 3. Populate Page 4 Experiment Questions Analysis Cards
  if (analysis_insights) {
    const q1 = document.getElementById('ans-q1');
    const q2 = document.getElementById('ans-q2');
    const q3 = document.getElementById('ans-q3');
    const q4 = document.getElementById('ans-q4');
    const q5 = document.getElementById('ans-q5');

    if (q1) q1.textContent = analysis_insights.question_1_system1_vs_system2 || '';
    if (q2) q2.textContent = analysis_insights.question_2_optimality_gap_and_success_rate || '';
    if (q3) q3.textContent = analysis_insights.question_3_malformed_response_rates || '';
    if (q4) q4.textContent = analysis_insights.question_4_exact_bfs_scalability || '';
    if (q5) q5.textContent = analysis_insights.question_5_robotics_deployment_recommendations || '';
  }
}
