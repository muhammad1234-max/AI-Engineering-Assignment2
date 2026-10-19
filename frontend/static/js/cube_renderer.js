/**
 * 3D Isometric Rubik's Cube Renderer for Clean Light Theme.
 * Maps standard 54-character URFDLB facelet strings into responsive SVG graphics.
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
 * Render a responsive 3D isometric projection showing U, F, and R faces.
 * ViewBox is tightly bounded (38 6 124 136) so it scales responsively
 * to fit any card container without vertical overflow.
 */
function renderIsometricCubeSVG(facelets) {
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

  // 1. UP (U) Face - Top isometric rhombus grid
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
        `<polygon points="${p0} ${p1} ${p2} ${p3}" fill="${fill}" stroke="#1E293B" stroke-width="1.8" stroke-linejoin="round" />`
      );
    }
  }

  // 2. FRONT (F) Face - Left vertical slanted grid
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
        `<polygon points="${p0} ${p1} ${p2} ${p3}" fill="${fill}" stroke="#1E293B" stroke-width="1.8" stroke-linejoin="round" />`
      );
    }
  }

  // 3. RIGHT (R) Face - Right vertical slanted grid
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
        `<polygon points="${p0} ${p1} ${p2} ${p3}" fill="${fill}" stroke="#1E293B" stroke-width="1.8" stroke-linejoin="round" />`
      );
    }
  }

  return `
    <svg viewBox="40 10 120 130" xmlns="http://www.w3.org/2000/svg" class="isometric-cube-svg" preserveAspectRatio="xMidYMid meet">
      <defs>
        <filter id="cube-subtle-shadow" x="-5%" y="-5%" width="115%" height="115%">
          <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#0F172A" flood-opacity="0.12"/>
        </filter>
      </defs>
      <g filter="url(#cube-subtle-shadow)">
        ${svgElements.join('\n')}
      </g>
    </svg>
  `;
}
