/**
 * 3D Isometric Rubik's Cube Renderer with Real-Time Facelet Animations.
 * Maps standard 54-character URFDLB facelet strings into responsive SVG graphics
 * and smoothly updates facelet colors and turn twists in real time during races.
 */

const FACE_COLORS = {
  U: '#FFFFFF', // White
  R: '#DC2626', // Red
  F: '#16A34A', // Green
  D: '#FACC15', // Yellow
  L: '#EA580C', // Orange
  B: '#2563EB', // Blue
};

// Subtle directional light shading for realistic 3D appearance
const FACE_SHADING = {
  U: 1.0,   // Top face (full illumination)
  F: 0.92,  // Front face (subtle front light)
  R: 0.82,  // Right face (slight shadow)
};

function shadeColor(hex, percent) {
  let num = parseInt(hex.slice(1), 16);
  let r = Math.min(255, Math.floor((num >> 16) * percent));
  let g = Math.min(255, Math.floor(((num >> 8) & 0x00ff) * percent));
  let b = Math.min(255, Math.floor((num & 0x0000ff) * percent));
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Generate SVG elements for the 3 visible faces (U, F, R).
 * Each polygon is given unique data-face and data-idx attributes
 * for instant in-place DOM updates and smooth color transitions.
 */
function buildCubePolygons(facelets) {
  if (!facelets || facelets.length !== 54) {
    facelets = 'U'.repeat(9) + 'R'.repeat(9) + 'F'.repeat(9) + 'D'.repeat(9) + 'L'.repeat(9) + 'B'.repeat(9);
  }

  const s = 20; // Edge size of one cubie
  const cos30 = Math.cos(Math.PI / 6);
  const sin30 = Math.sin(Math.PI / 6);
  const dx = s * cos30; // ~17.32
  const dy = s * sin30; // 10.0

  const cx = 100;
  const cy = 15;

  let svgElements = [];

  const uFace = facelets.slice(0, 9);
  const rFace = facelets.slice(9, 18);
  const fFace = facelets.slice(18, 27);

  // 1. UP (U) Face - Top isometric rhombus grid (Indices 0..8)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const idx = r * 3 + c;
      const colorChar = uFace[idx] || 'U';
      const baseColor = FACE_COLORS[colorChar] || '#FFFFFF';
      const fill = shadeColor(baseColor, FACE_SHADING.U);

      const p0 = `${(cx + (c - r) * dx).toFixed(1)},${(cy + (c + r) * dy).toFixed(1)}`;
      const p1 = `${(cx + (c - r + 1) * dx).toFixed(1)},${(cy + (c + r + 1) * dy).toFixed(1)}`;
      const p2 = `${(cx + (c - r) * dx).toFixed(1)},${(cy + (c + r + 2) * dy).toFixed(1)}`;
      const p3 = `${(cx + (c - r - 1) * dx).toFixed(1)},${(cy + (c + r + 1) * dy).toFixed(1)}`;

      svgElements.push(
        `<polygon class="facelet-poly" data-face="U" data-idx="${idx}" points="${p0} ${p1} ${p2} ${p3}" fill="${fill}" style="fill: ${fill};" stroke="#1E293B" stroke-width="1.8" stroke-linejoin="round" />`
      );
    }
  }

  // 2. FRONT (F) Face - Left vertical slanted grid (Indices 0..8)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const idx = r * 3 + c;
      const colorChar = fFace[idx] || 'F';
      const baseColor = FACE_COLORS[colorChar] || '#16A34A';
      const fill = shadeColor(baseColor, FACE_SHADING.F);

      const p0 = `${(cx - (3 - c) * dx).toFixed(1)},${(cy + (3 + c) * dy + r * s).toFixed(1)}`;
      const p1 = `${(cx - (2 - c) * dx).toFixed(1)},${(cy + (4 + c) * dy + r * s).toFixed(1)}`;
      const p2 = `${(cx - (2 - c) * dx).toFixed(1)},${(cy + (4 + c) * dy + (r + 1) * s).toFixed(1)}`;
      const p3 = `${(cx - (3 - c) * dx).toFixed(1)},${(cy + (3 + c) * dy + (r + 1) * s).toFixed(1)}`;

      svgElements.push(
        `<polygon class="facelet-poly" data-face="F" data-idx="${idx}" points="${p0} ${p1} ${p2} ${p3}" fill="${fill}" style="fill: ${fill};" stroke="#1E293B" stroke-width="1.8" stroke-linejoin="round" />`
      );
    }
  }

  // 3. RIGHT (R) Face - Right vertical slanted grid (Indices 0..8)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const idx = r * 3 + c;
      const colorChar = rFace[idx] || 'R';
      const baseColor = FACE_COLORS[colorChar] || '#DC2626';
      const fill = shadeColor(baseColor, FACE_SHADING.R);

      const p0 = `${(cx + c * dx).toFixed(1)},${(cy + (6 - c) * dy + r * s).toFixed(1)}`;
      const p1 = `${(cx + (c + 1) * dx).toFixed(1)},${(cy + (5 - c) * dy + r * s).toFixed(1)}`;
      const p2 = `${(cx + (c + 1) * dx).toFixed(1)},${(cy + (5 - c) * dy + (r + 1) * s).toFixed(1)}`;
      const p3 = `${(cx + c * dx).toFixed(1)},${(cy + (6 - c) * dy + (r + 1) * s).toFixed(1)}`;

      svgElements.push(
        `<polygon class="facelet-poly" data-face="R" data-idx="${idx}" points="${p0} ${p1} ${p2} ${p3}" fill="${fill}" style="fill: ${fill};" stroke="#1E293B" stroke-width="1.8" stroke-linejoin="round" />`
      );
    }
  }

  return svgElements.join('\n');
}

/**
 * Generate full SVG markup with subtle drop-shadow and floating move badge.
 */
function renderIsometricCubeSVG(facelets, lastMove = '') {
  return `
    <div class="cube-move-pill ${lastMove ? 'active' : ''}">${lastMove || ''}</div>
    <svg viewBox="40 10 120 130" xmlns="http://www.w3.org/2000/svg" class="isometric-cube-svg" preserveAspectRatio="xMidYMid meet">
      <defs>
        <filter id="cube-subtle-shadow" x="-5%" y="-5%" width="115%" height="115%">
          <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#0F172A" flood-opacity="0.12"/>
        </filter>
      </defs>
      <g filter="url(#cube-subtle-shadow)" class="cube-polys-group">
        ${buildCubePolygons(facelets)}
      </g>
    </svg>
  `;
}

/**
 * Real-Time Animated Cube Updater.
 * Updates polygon fills in-place to trigger CSS transition animations,
 * applies a tactile turn rotation twist, and pops the floating move badge.
 */
function updateIsometricCube(containerId, facelets, lastMove = null) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const svg = container.querySelector('svg.isometric-cube-svg');
  if (!svg) {
    container.innerHTML = renderIsometricCubeSVG(facelets, lastMove);
    return;
  }

  if (!facelets || facelets.length !== 54) return;

  const uFace = facelets.slice(0, 9);
  const rFace = facelets.slice(9, 18);
  const fFace = facelets.slice(18, 27);

  // In-place smooth color updates
  const polys = container.querySelectorAll('.facelet-poly');
  polys.forEach((poly) => {
    const face = poly.getAttribute('data-face');
    const idx = parseInt(poly.getAttribute('data-idx'), 10);
    let colorChar = 'U';
    let shading = 1.0;

    if (face === 'U') {
      colorChar = uFace[idx] || 'U';
      shading = FACE_SHADING.U;
    } else if (face === 'F') {
      colorChar = fFace[idx] || 'F';
      shading = FACE_SHADING.F;
    } else if (face === 'R') {
      colorChar = rFace[idx] || 'R';
      shading = FACE_SHADING.R;
    }

    const baseColor = FACE_COLORS[colorChar] || '#FFFFFF';
    const newFill = shadeColor(baseColor, shading);
    
    // Set both style.fill (for CSS transition) and attribute (for standard SVG)
    poly.style.fill = newFill;
    poly.setAttribute('fill', newFill);
  });

  // Animate turn twist and move indicator pill if a move was executed
  if (lastMove) {
    svg.classList.remove('cube-turn-anim');
    void svg.offsetWidth; // Force DOM reflow to re-trigger keyframe
    svg.classList.add('cube-turn-anim');

    let pill = container.querySelector('.cube-move-pill');
    if (!pill) {
      pill = document.createElement('div');
      pill.className = 'cube-move-pill';
      container.prepend(pill);
    }

    pill.textContent = lastMove;
    pill.classList.remove('active');
    void pill.offsetWidth;
    pill.classList.add('active');

    if (pill._clearTimer) clearTimeout(pill._clearTimer);
    pill._clearTimer = setTimeout(() => {
      pill.classList.remove('active');
    }, 850);
  }
}
