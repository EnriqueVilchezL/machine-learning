/* Shared runtime for the Article Slides decks: helpers for drawing SVG figures,
   slide headers, admonition icons, KaTeX, step-by-step fragments and slide activation.
   Load after deck-stage.js; deck-specific scripts read the helpers from window.Deck. */
(() => {
'use strict';
const NS = 'http://www.w3.org/2000/svg';
const deck = document.querySelector('deck-stage');
const SLIDES = [...deck.children].filter((n) => n.tagName === 'SECTION');
const SHOW_ALL = new URLSearchParams(location.search).has('all');
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const VENUE = 'Machine Learning';

/* ───────── helpers ───────── */
function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}
function rng(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  let u = 0, v = 0;
  while (!u) u = r();
  while (!v) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
function fmt(v, digits = 2) {
  if (!isFinite(v)) return '∞';
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString('en-US').replace(/,/g, ' ');
  return v.toFixed(digits);
}
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}
function polyfit(xs, ys, deg, ridge = 1e-12) {
  const m = deg + 1;
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const b = new Array(m).fill(0);
  xs.forEach((x, i) => {
    const pw = Array.from({ length: m }, (_, k) => x ** k);
    for (let r = 0; r < m; r++) {
      b[r] += pw[r] * ys[i];
      for (let c = 0; c < m; c++) A[r][c] += pw[r] * pw[c];
    }
  });
  for (let r = 0; r < m; r++) A[r][r] += ridge;
  return solve(A, b);
}
const polyval = (c, x) => c.reduceRight((acc, ck) => acc * x + ck, 0);
function curvePath(f, x0, x1, sx, sy, n = 240) {
  let d = '';
  for (let i = 0; i <= n; i++) {
    const x = x0 + ((x1 - x0) * i) / n;
    d += (i ? 'L' : 'M') + sx(x).toFixed(1) + ' ' + clamp(sy(f(x)), -5000, 5000).toFixed(1);
  }
  return d;
}
function toSvgPoint(svg, ev) {
  const p = svg.createSVGPoint();
  p.x = ev.clientX; p.y = ev.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}
/* Axes, grid and a clipped plot group in user units of the svg viewBox */
function frame(svg, o) {
  const W = o.W, H = o.H, L = o.l ?? 96, R = o.r ?? 24, T = o.t ?? 24, B = o.b ?? 84;
  const sx = lin(o.x[0], o.x[1], L, W - R), sy = lin(o.y[0], o.y[1], H - B, T);
  const back = el('g', {}, svg);
  const fx = o.fx || String, fy = o.fy || String;
  (o.xt || []).forEach((v) => {
    if (o.grid !== false) el('line', { class: 'gl', x1: sx(v), x2: sx(v), y1: T, y2: H - B }, back);
    el('text', { class: 'tk', x: sx(v), y: H - B + 32, 'text-anchor': 'middle', text: fx(v) }, back);
  });
  (o.yt || []).forEach((v) => {
    if (o.grid !== false) el('line', { class: 'gl', x1: L, x2: W - R, y1: sy(v), y2: sy(v) }, back);
    el('text', { class: 'tk', x: L - 12, y: sy(v) + 8, 'text-anchor': 'end', text: fy(v) }, back);
  });
  el('path', { class: 'ax', d: `M${L} ${T}V${H - B}H${W - R}` }, back);
  if (o.xl) el('text', { class: 'lb', x: (L + W - R) / 2, y: H - B + 72, 'text-anchor': 'middle', text: o.xl }, back);
  if (o.yl) el('text', { class: 'lb', transform: `translate(${L - 66} ${(T + H - B) / 2}) rotate(-90)`, 'text-anchor': 'middle', text: o.yl }, back);
  const id = svg.id + '-clip';
  const cp = el('clipPath', { id }, svg);
  el('rect', { x: L, y: T, width: W - R - L, height: H - B - T }, cp);
  const plot = el('g', { 'clip-path': `url(#${id})` }, svg);
  return { sx, sy, plot, back, L, R, T, B, W, H, ix: lin(L, W - R, o.x[0], o.x[1]), iy: lin(H - B, T, o.y[0], o.y[1]) };
}
function legend(svg, x, y, items) {
  const g = el('g', {}, svg);
  items.forEach(([kind, cls, label], i) => {
    const yy = y + i * 36;
    if (kind === 'dot') el('circle', { class: cls, cx: x + 12, cy: yy, r: 9 }, g);
    else if (kind === 'ring') el('circle', { cx: x + 12, cy: yy, r: 8, style: 'fill:var(--myst-color-bg);stroke:var(--c-b);stroke-width:3' }, g);
    else el('line', { class: cls, x1: x, x2: x + 26, y1: yy, y2: yy }, g);
    el('text', { class: 'txs', x: x + 38, y: yy + 8, text: label }, g);
  });
  return g;
}

/* ───────── slide headers, icons, math ───────── */
SLIDES.forEach((s, i) => {
  const h = s.querySelector('.hdr');
  if (!h) return;
  h.innerHTML = `<div class="subj">${h.dataset.s}</div><div class="venue">${VENUE}</div><div class="grow"></div><div class="num">${i + 1} / ${SLIDES.length}</div>`;
});
const ICON_PATHS = {
  note: '<path stroke-linecap="round" stroke-linejoin="round" d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"/>',
  tip: '<path stroke-linecap="round" stroke-linejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10"/>',
  hint: '<path stroke-linecap="round" stroke-linejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 0 0 1.5-.189m-1.5.189a6.01 6.01 0 0 1-1.5-.189m3.75 7.478a12.06 12.06 0 0 1-4.5 0m3.75 2.383a14.406 14.406 0 0 1-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 1 0-7.517 0c.85.493 1.509 1.333 1.509 2.316V18"/>',
  caution: '<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"/>',
  attention: '<path stroke-linecap="round" stroke-linejoin="round" d="M10.34 15.84c-.688-.06-1.386-.09-2.09-.09H7.5a4.5 4.5 0 1 1 0-9h.75c.704 0 1.402-.03 2.09-.09m0 9.18c.253.962.584 1.892.985 2.783.247.55.06 1.21-.463 1.511l-.657.38c-.551.318-1.26.117-1.527-.461a20.845 20.845 0 0 1-1.44-4.282m3.102.069a18.03 18.03 0 0 1-.59-4.59c0-1.586.205-3.124.59-4.59m0 9.18a23.848 23.848 0 0 1 8.835 2.535M10.34 6.66a23.847 23.847 0 0 0 8.835-2.535m0 0A23.74 23.74 0 0 0 18.795 3m.38 1.125a23.91 23.91 0 0 1 1.014 5.395m-1.014 8.855c-.118.38-.245.754-.38 1.125m.38-1.125a23.91 23.91 0 0 0 1.014-5.395m0-3.46c.495.413.811 1.035.811 1.73 0 .695-.316 1.317-.811 1.73m0-3.46a24.347 24.347 0 0 1 0 3.46"/>',
  warning: '<path fill="currentColor" stroke="none" fill-rule="evenodd" clip-rule="evenodd" d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"/>',
  danger: '<path fill="currentColor" stroke="none" fill-rule="evenodd" clip-rule="evenodd" d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"/>',
};
document.querySelectorAll('i[data-i]').forEach((i) => {
  i.outerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">${ICON_PATHS[i.dataset.i]}</svg>`;
});
let mathTries = 0;
function renderMath() {
  if (!window.katex) { if (mathTries++ < 80) setTimeout(renderMath, 100); return; }
  document.querySelectorAll('.tex').forEach((e) => katex.render(e.textContent, e, { throwOnError: false }));
  document.querySelectorAll('.tex-d').forEach((e) => katex.render(e.textContent, e, { throwOnError: false, displayMode: true }));
}
renderMath();

/* ───────── fragments: → reveals the next step before leaving the slide ───────── */
function stepsOf(slide) {
  const groups = [], byKey = new Map();
  slide.querySelectorAll('.frag').forEach((f, i) => {
    const key = f.dataset.step || '_' + i;
    if (!byKey.has(key)) { byKey.set(key, []); groups.push(byKey.get(key)); }
    byKey.get(key).push(f);
  });
  return groups;
}
let current = null;
const FORWARD = new Set(['ArrowRight', 'PageDown', ' ', 'Spacebar', 'ArrowDown']);
const BACKWARD = new Set(['ArrowLeft', 'PageUp', 'ArrowUp']);
window.addEventListener('keydown', (e) => {
  const t = e.composedPath ? e.composedPath()[0] : e.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'd' || e.key === 'D') { document.documentElement.classList.toggle('dark'); return; }
  if (!current) return;
  const steps = stepsOf(current);
  let hit = null;
  if (FORWARD.has(e.key)) {
    hit = steps.find((g) => g.some((f) => !f.classList.contains('on')));
    if (hit) hit.forEach((f) => f.classList.add('on'));
  } else if (BACKWARD.has(e.key)) {
    hit = [...steps].reverse().find((g) => g.some((f) => f.classList.contains('on')));
    if (hit) hit.forEach((f) => f.classList.remove('on'));
  }
  if (hit) { e.preventDefault(); e.stopImmediatePropagation(); }
}, true);


/* ───────── slide activation ───────── */
function activate(slide, backwards) {
  if (!slide) return;
  current = slide;
  stepsOf(slide).forEach((g) => g.forEach((f) => f.classList.toggle('on', backwards || SHOW_ALL)));
  slide.querySelectorAll('svg').forEach((v) => {
    if (v.ownerSVGElement) return;
    try { v.setCurrentTime(0); if (REDUCED) v.pauseAnimations(); } catch (err) { /* not an animated root */ }
  });
}
deck.addEventListener('slidechange', (e) => activate(e.detail.slide, e.detail.previousIndex > e.detail.index));
// Deck-specific scripts may add fragments after this file runs, so the first activation waits for them.
window.addEventListener('DOMContentLoaded', () => activate(SLIDES.find((s) => s.hasAttribute('data-deck-active')) || SLIDES[0], false));

window.Deck = { NS, REDUCED, el, rng, gauss, lin, clamp, mean, fmt, solve, polyfit, polyval, curvePath, toSvgPoint, frame, legend };
})();
