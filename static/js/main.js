/**
 * main.js
 * =======
 * Shared utilities for Tourism Trend Analytics Flask App.
 *
 * Exports (global):
 *   fetchJSON(url)          — fetch JSON from API endpoint
 *   setText(id, value)      — set element inner text
 *   fmt(n)                  — format number with commas
 *   loading(id, spin, msg)  — show loading/error in chart container
 *   plotlyCfg()             — Plotly config object
 *   plotlyLayout            — Plotly layout factory function
 *   PALETTE_PURP            — Purple gradient palette
 *   PALETTE_TEAL            — Teal gradient palette
 *   PALETTE_GREEN           — Green gradient palette
 *   PALETTE_SEQ             — Sequential palette for rating bars
 */

'use strict';

// ── Colour Palettes ───────────────────────────────────────────────────────────
const PALETTE_PURP  = [
  '#3b0764','#4c1d95','#5b21b6','#6d28d9','#7c3aed',
  '#8b5cf6','#a78bfa','#c4b5fd','#ddd6fe','#ede9fe',
];
const PALETTE_TEAL  = [
  '#134e4a','#115e59','#0f766e','#0d9488','#14b8a6',
  '#2dd4bf','#5eead4','#99f6e4','#ccfbf1','#f0fdfa',
];
const PALETTE_GREEN = [
  '#064e3b','#065f46','#047857','#059669','#10b981',
  '#34d399','#6ee7b7','#a7f3d0','#d1fae5','#ecfdf5',
];
const PALETTE_SEQ   = ['#f87171','#fb923c','#fbbf24','#4ade80','#34d399'];

// ── Plotly shared config ──────────────────────────────────────────────────────
function plotlyCfg() {
  return {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'tourism_analytics_chart',
      height: 600, width: 1200, scale: 2,
    },
  };
}

// ── Plotly layout factory ─────────────────────────────────────────────────────
/**
 * @param {string} title
 * @param {number} height
 * @param {Object} overrides  e.g. { yaxis_tickfont: 9, xaxis_range: [0,1] }
 */
function plotlyLayout(title = '', height = 380, overrides = {}) {
  const base = {
    title: {
      text: title,
      font: { size: 14, color: '#c4b5fd', family: 'Inter' },
      x: 0.02,
    },
    height,
    paper_bgcolor: 'rgba(255,255,255,0.0)',
    plot_bgcolor:  'rgba(255,255,255,0.0)',
    font: { family: 'Inter', color: '#c0bce8', size: 12 },
    legend: {
      bgcolor: 'rgba(255,255,255,0.04)',
      bordercolor: 'rgba(255,255,255,0.1)',
      borderwidth: 1,
      font: { color: '#c0bce8' },
    },
    margin: { l: 150, r: 30, t: 48, b: 45 },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.06)',
      zerolinecolor: 'rgba(255,255,255,0.08)',
      tickfont: { color: '#9090b8' },
    },
    yaxis: {
      gridcolor: 'rgba(255,255,255,0.06)',
      zerolinecolor: 'rgba(255,255,255,0.08)',
      tickfont: { color: '#9090b8' },
    },
  };

  // Apply overrides (supports dotted keys like yaxis_tickfont)
  for (const [key, val] of Object.entries(overrides)) {
    if (key.includes('_')) {
      const [axis, prop] = key.split('_');
      if (!base[axis]) base[axis] = {};
      if (prop === 'tickfont') {
        base[axis].tickfont = { ...(base[axis].tickfont || {}), size: val };
      } else if (prop === 'tickangle') {
        base[axis].tickangle = val;
      } else if (prop === 'range') {
        base[axis].range = val;
      } else {
        base[axis][prop] = val;
      }
    } else {
      base[key] = val;
    }
  }
  return base;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
/**
 * Fetch JSON from a local API endpoint with error handling.
 * @param {string} url
 * @returns {Promise<Object|null>}
 */
async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error(`fetchJSON(${url}) failed:`, err);
    return null;
  }
}

/**
 * Set element inner text safely.
 * @param {string} id
 * @param {*} value
 */
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? '—';
}

/**
 * Format a number with thousands separators.
 * @param {number|null} n
 * @returns {string}
 */
function fmt(n) {
  if (n == null || n === undefined || isNaN(n)) return '—';
  return Number(n).toLocaleString('en-IN');
}

/**
 * Show a loading spinner or error message inside a chart container.
 * @param {string} id       element ID
 * @param {boolean} spin    true = spinner, false = message
 * @param {string} msg      message to show when spin=false
 */
function loading(id, spin = true, msg = 'No data available') {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = spin
    ? `<div class="loading-overlay"><div class="spinner"></div><div class="loading-text">Loading…</div></div>`
    : `<div class="loading-overlay" style="opacity:0.6;"><span style="font-size:1.2rem;">📭</span><div class="loading-text">${msg}</div></div>`;
}

// ── Theme Toggle ──────────────────────────────────────────────────────────────
(function initTheme() {
  const btn       = document.getElementById('themeToggle');
  const icon      = document.getElementById('themeIcon');
  const label     = document.getElementById('themeLabel');
  const htmlEl    = document.documentElement;
  const STORAGE   = 'bdt_theme';

  function applyTheme(theme) {
    htmlEl.setAttribute('data-theme', theme);
    if (theme === 'light') {
      if (icon)  icon.textContent  = '🌙';
      if (label) label.textContent = 'Dark';
    } else {
      if (icon)  icon.textContent  = '☀️';
      if (label) label.textContent = 'Light';
    }
    localStorage.setItem(STORAGE, theme);

    // Update Plotly paper/plot colours after theme switch (re-layout all charts)
    setTimeout(() => {
      const plots = document.querySelectorAll('.js-plotly-plot');
      plots.forEach(p => {
        try {
          Plotly.relayout(p, {
            paper_bgcolor: 'rgba(255,255,255,0.0)',
            plot_bgcolor:  'rgba(255,255,255,0.0)',
          });
        } catch (_) {}
      });
    }, 200);
  }

  // Restore saved preference
  const saved = localStorage.getItem(STORAGE) || 'dark';
  applyTheme(saved);

  if (btn) {
    btn.addEventListener('click', () => {
      const next = htmlEl.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(next);
    });
  }
})();

// ── Sidebar Toggle ────────────────────────────────────────────────────────────
(function initSidebar() {
  const toggleBtn  = document.getElementById('sidebarToggle');
  const sidebar    = document.getElementById('sidebar');
  const mainContent = document.getElementById('mainContent');

  if (!toggleBtn || !sidebar) return;

  toggleBtn.addEventListener('click', () => {
    const isMobile = window.innerWidth <= 768;
    if (isMobile) {
      sidebar.classList.toggle('open');
    } else {
      sidebar.classList.toggle('collapsed');
      mainContent && mainContent.classList.toggle('expanded');
    }
  });

  // Close sidebar on mobile when clicking outside
  document.addEventListener('click', e => {
    if (window.innerWidth <= 768 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== toggleBtn) {
      sidebar.classList.remove('open');
    }
  });
})();

// ── Active nav highlight ──────────────────────────────────────────────────────
(function highlightNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
    const href = link.getAttribute('href');
    if (href && (path === href || (path === '/' && href === '/dashboard'))) {
      link.classList.add('active');
    }
  });
})();

// ── Animate KPI card counters ─────────────────────────────────────────────────
function animateCounter(id, target, duration = 1200, isFloat = false) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = Date.now();
  const from  = 0;

  function tick() {
    const elapsed = Date.now() - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (target - from) * eased;
    el.textContent = isFloat
      ? current.toFixed(2)
      : Math.round(current).toLocaleString('en-IN');
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
