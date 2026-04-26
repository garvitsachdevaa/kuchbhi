"""
Animated robot orchestrator widget for the SpindleFlow RL demo.
Exports one public function: render_orchestrator(state, height=600)

All HTML/CSS/JS is self-contained — no CDN, no external calls.
Safe for Hugging Face Spaces iframe sandbox.
"""

from __future__ import annotations
import json
import math

# ── Agent color and icon maps ─────────────────────────────────────────────────

SPEC_COLORS = {
    "frontend_react":     "#00d4ff",
    "backend_api":        "#7c3aed",
    "database_architect": "#f59e0b",
    "devops_engineer":    "#10b981",
    "security_analyst":   "#ef4444",
    "product_strategist": "#8b5cf6",
    "ux_designer":        "#ec4899",
    "tech_writer":        "#94a3b8",
}

SPEC_ICONS = {
    "frontend_react":     "FE",
    "backend_api":        "API",
    "database_architect": "DB",
    "devops_engineer":    "OPS",
    "security_analyst":   "SEC",
    "product_strategist": "PM",
    "ux_designer":        "UX",
    "tech_writer":        "DOC",
}

_SPAWNED_COLOR   = "#fbbf24"   # gold for auto-spawned agents
_FALLBACK_COLORS = [           # cycle through for multiple unknown agents
    "#fbbf24", "#f472b6", "#34d399", "#fb923c", "#a78bfa",
]


def _agent_color(agent_id: str, spawned_ids: set) -> str:
    if agent_id in SPEC_COLORS:
        return SPEC_COLORS[agent_id]
    if agent_id in spawned_ids:
        return _SPAWNED_COLOR
    # deterministic fallback based on hash
    return _FALLBACK_COLORS[hash(agent_id) % len(_FALLBACK_COLORS)]


def _agent_icon(agent_id: str, spawned_ids: set) -> str:
    if agent_id in SPEC_ICONS:
        return SPEC_ICONS[agent_id]
    if agent_id in spawned_ids:
        return "⚡"
    return agent_id[:3].upper()


# ── Layout ────────────────────────────────────────────────────────────────────

def _agent_positions(agent_ids: list,
                     canvas_w: int = 780,
                     canvas_h: int = 560) -> dict:
    """Return {agent_id: (x, y)} in a straight vertical column on the right."""
    col_x = canvas_w - 115
    n = len(agent_ids)
    if n == 0:
        return {}
    pad_top = 50
    pad_bot = 50
    usable  = canvas_h - pad_top - pad_bot
    step    = usable / n
    positions = {}
    for i, aid in enumerate(agent_ids):
        y = round(pad_top + step * i + step / 2)
        positions[aid] = (col_x, y)
    return positions


# ── SVG builders ──────────────────────────────────────────────────────────────

def _robot_svg() -> str:
    return """
    <g id="robot" transform="translate(160, 280)">

      <!-- Antenna -->
      <line x1="0" y1="-115" x2="0" y2="-95" stroke="#00d4ff" stroke-width="2"/>
      <circle cx="0" cy="-120" r="5" fill="#00d4ff" class="antenna-pulse"/>

      <!-- Head -->
      <rect x="-38" y="-95" width="76" height="62" rx="10"
            fill="#0d1117" stroke="#00d4ff" stroke-width="1.5"
            class="head-glow"/>

      <!-- Left Eye -->
      <circle cx="-14" cy="-68" r="10" fill="#001a2e"/>
      <circle cx="-14" cy="-68" r="6" fill="#00d4ff" class="eye-left"/>
      <circle cx="-11" cy="-71" r="2" fill="white" opacity="0.6"/>

      <!-- Right Eye -->
      <circle cx="14" cy="-68" r="10" fill="#001a2e"/>
      <circle cx="14" cy="-68" r="6" fill="#00d4ff" class="eye-right"/>
      <circle cx="17" cy="-71" r="2" fill="white" opacity="0.6"/>

      <!-- Mouth -->
      <path d="M -14 -46 Q 0 -38 14 -46"
            fill="none" stroke="#00d4ff" stroke-width="2"
            stroke-linecap="round" class="mouth"/>

      <!-- Neck -->
      <rect x="-8" y="-33" width="16" height="10" rx="3"
            fill="#0d1117" stroke="#1a2a3a" stroke-width="1"/>

      <!-- Body -->
      <rect x="-45" y="-23" width="90" height="80" rx="12"
            fill="#0a0f1a" stroke="#1a3a5a" stroke-width="1.5"/>

      <!-- Core (spinning hexagon) -->
      <g class="core-spin" transform="translate(0, 17)">
        <polygon points="0,-18 15.6,-9 15.6,9 0,18 -15.6,9 -15.6,-9"
                 fill="none" stroke="#00d4ff" stroke-width="1.5" opacity="0.8"/>
        <polygon points="0,-11 9.5,-5.5 9.5,5.5 0,11 -9.5,5.5 -9.5,-5.5"
                 fill="rgba(0,212,255,0.15)" stroke="#00d4ff" stroke-width="1"/>
        <circle cx="0" cy="0" r="4" fill="#00d4ff" class="core-pulse"/>
      </g>

      <!-- Left Arm -->
      <g id="arm-left">
        <rect x="-68" y="-18" width="24" height="12" rx="6"
              fill="#0a0f1a" stroke="#1a3a5a" stroke-width="1.5"/>
        <rect x="-72" y="-8" width="14" height="28" rx="7"
              fill="#0a0f1a" stroke="#1a3a5a" stroke-width="1.5"/>
      </g>

      <!-- Right Arm -->
      <g id="arm-right" class="arm-idle">
        <rect x="44" y="-18" width="24" height="12" rx="6"
              fill="#0a0f1a" stroke="#00d4ff" stroke-width="1.5"/>
        <rect x="58" y="-8" width="14" height="28" rx="7"
              fill="#0a0f1a" stroke="#00d4ff" stroke-width="1.5"/>
        <circle cx="65" cy="22" r="5" fill="#00d4ff" class="hand-glow"/>
      </g>

      <!-- Legs -->
      <rect x="-28" y="57" width="18" height="28" rx="6"
            fill="#0a0f1a" stroke="#1a3a5a" stroke-width="1.5"/>
      <rect x="10" y="57" width="18" height="28" rx="6"
            fill="#0a0f1a" stroke="#1a3a5a" stroke-width="1.5"/>

      <!-- Feet -->
      <ellipse cx="-19" cy="87" rx="16" ry="7"
               fill="#0a0f1a" stroke="#1a3a5a" stroke-width="1"/>
      <ellipse cx="19" cy="87" rx="16" ry="7"
               fill="#0a0f1a" stroke="#1a3a5a" stroke-width="1"/>

      <!-- Shadow -->
      <ellipse cx="0" cy="97" rx="50" ry="8"
               fill="rgba(0,212,255,0.05)"/>
    </g>
    """


def _agent_card_svg(agent_id: str, x: int, y: int,
                    status: str, color: str,
                    is_spawned: bool = False) -> str:
    """Returns SVG <g> for one agent card. status: idle | active | done."""
    icon  = SPEC_ICONS.get(agent_id, ("⚡" if is_spawned else agent_id[:3].upper()))
    label = agent_id.replace("_", " ").title()
    label = label[:18] + ("…" if len(label) > 18 else "")

    status_class = {"idle": "agent-idle", "active": "agent-active",
                    "done": "agent-done"}.get(status, "agent-idle")
    opacity = "1.0" if status != "idle" else "0.40"
    border  = "#fbbf24" if is_spawned else color
    spawn_star = (
        f'<text x="26" y="-26" text-anchor="middle" font-size="10" fill="#fbbf24">⚡</text>'
        if is_spawned else ""
    )

    return f"""
    <g class="agent-card {status_class}" transform="translate({x},{y})"
       id="agent-{agent_id}" opacity="{opacity}">
      <circle cx="0" cy="0" r="36" fill="none"
              stroke="{border}" stroke-width="1.5"
              class="agent-ring" opacity="0.25"/>
      <rect x="-26" y="-26" width="52" height="52" rx="10"
            fill="#0a0f1a" stroke="{border}" stroke-width="1.5"
            opacity="0.95"/>
      <text x="0" y="5" text-anchor="middle" dominant-baseline="middle"
            fill="{color}" font-family="'JetBrains Mono', monospace"
            font-size="11" font-weight="700">{icon}</text>
      {spawn_star}
      <circle cx="20" cy="-20" r="5" fill="{color}" class="status-dot"/>
      <text x="0" y="40" text-anchor="middle"
            fill="#64748b" font-family="system-ui, sans-serif"
            font-size="8.5" letter-spacing="0.3">{label}</text>
      <g class="done-check" opacity="0">
        <circle cx="20" cy="-20" r="7" fill="#10b981"/>
        <text x="20" y="-16" text-anchor="middle" fill="white" font-size="9">✓</text>
      </g>
    </g>
    """


def _beam_svg(edges: list, agent_positions: dict) -> str:
    """Returns SVG beam lines for all current delegation edges."""
    robot_hand_x, robot_hand_y = 225, 302
    lines = []
    for caller, callee in edges:
        if callee not in agent_positions:
            continue
        tx, ty = agent_positions[callee]
        color  = SPEC_COLORS.get(callee, _SPAWNED_COLOR)
        lines.append(f"""
        <line id="beam-{callee}"
              x1="{robot_hand_x}" y1="{robot_hand_y}" x2="{tx}" y2="{ty}"
              stroke="{color}" stroke-width="1.5" stroke-linecap="round"
              opacity="0.55" stroke-dasharray="6 4" class="beam-line beam-animate"/>
        <circle id="dot-{callee}" r="4" fill="{color}" opacity="0.9" class="beam-dot">
          <animateMotion dur="0.9s" repeatCount="indefinite"
                         path="M {robot_hand_x},{robot_hand_y} L {tx},{ty}"/>
        </circle>
        <circle id="burst-{callee}" cx="{tx}" cy="{ty}" r="8"
                fill="none" stroke="{color}" stroke-width="2"
                opacity="0" class="burst-ring burst-animate"/>
        """)
    return "\n".join(lines)


# ── HTML template ─────────────────────────────────────────────────────────────

def _html_template(*, agents_svg, beams_svg, robot_svg, state_json,
                   task_short, reward_html, step, phase, mode, mode_color) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'JetBrains Mono', 'Fira Code', monospace; overflow: hidden; }}

  .canvas-wrap {{
    position: relative; width: 100%; height: 560px;
    background: radial-gradient(ellipse at 25% 50%, rgba(0,212,255,0.04) 0%, transparent 60%),
                radial-gradient(ellipse at 85% 50%, rgba(124,58,237,0.03) 0%, transparent 50%),
                #080d14;
    border-radius: 16px; border: 1px solid rgba(0,212,255,0.1); overflow: hidden;
  }}
  .canvas-wrap::before {{
    content: ''; position: absolute; inset: 0;
    background-image: linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px; border-radius: 16px; pointer-events: none;
  }}
  svg.main-svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}

  .info-bar {{
    position: absolute; bottom: 0; left: 0; right: 0; height: 44px;
    background: rgba(0,0,0,0.5); border-top: 1px solid rgba(255,255,255,0.05);
    border-radius: 0 0 16px 16px; display: flex; align-items: center;
    padding: 0 20px; gap: 24px; font-size: 11px; color: #475569;
  }}
  .info-badge {{ display: flex; align-items: center; gap: 6px; }}
  .info-badge .label {{ font-size: 9px; text-transform: uppercase; letter-spacing: 1px; color: #334155; }}
  .info-badge .value {{ font-weight: 700; color: #94a3b8; }}
  .task-text {{ flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; color: #475569; font-size: 10px; }}

  .orch-label   {{ position: absolute; top: 18px; left: 18px; font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #00d4ff; opacity: 0.7; }}
  .agents-label {{ position: absolute; top: 18px; right: 18px; font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #475569; opacity: 0.7; }}

  .divider-line {{
    position: absolute; left: 47%; top: 8%; height: 84%; width: 1px;
    background: linear-gradient(to bottom, transparent, rgba(0,212,255,0.12), transparent);
  }}

  /* Robot animations */
  @keyframes antenna-blink {{ 0%,90%,100% {{ opacity:1; }} 95% {{ opacity:0.2; }} }}
  .antenna-pulse {{ animation: antenna-blink 2.5s ease-in-out infinite; }}

  @keyframes core-rotation {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
  .core-spin {{ transform-origin: 0px 17px; animation: core-rotation 4s linear infinite; }}

  @keyframes core-pulse {{ 0%,100% {{ opacity:0.8; r:4px; }} 50% {{ opacity:1; r:6px; fill:white; }} }}
  .core-pulse {{ animation: core-pulse 1.5s ease-in-out infinite; }}

  @keyframes eye-blink {{ 0%,92%,100% {{ ry:6px; }} 96% {{ ry:1px; }} }}
  .eye-left, .eye-right {{ animation: eye-blink 4s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}

  @keyframes hand-glow {{ 0%,100% {{ opacity:0.6; r:5px; }} 50% {{ opacity:1; r:8px; }} }}
  .hand-glow {{ animation: hand-glow 1.2s ease-in-out infinite; }}

  @keyframes head-glow-pulse {{ 0%,100% {{ filter: drop-shadow(0 0 4px rgba(0,212,255,0.3)); }} 50% {{ filter: drop-shadow(0 0 12px rgba(0,212,255,0.7)); }} }}
  .head-glow {{ animation: head-glow-pulse 2s ease-in-out infinite; }}

  @keyframes arm-extend {{ 0% {{ transform: rotate(0deg) translateX(0px); }} 100% {{ transform: rotate(-15deg) translateX(12px); }} }}
  .arm-delegating {{ transform-origin: 55px 0px; animation: arm-extend 0.4s ease-out forwards; }}

  /* Agent animations */
  @keyframes agent-active-pulse {{ 0%,100% {{ filter: drop-shadow(0 0 6px currentColor); }} 50% {{ filter: drop-shadow(0 0 18px currentColor); }} }}
  .agent-active {{ animation: agent-active-pulse 0.8s ease-in-out infinite; opacity: 1 !important; }}
  .agent-done {{ opacity: 1 !important; }}
  .agent-done .status-dot {{ fill: #10b981 !important; }}
  .agent-done .done-check {{ opacity: 1 !important; }}

  @keyframes ring-expand {{ from {{ r:28px; opacity:0.6; }} to {{ r:48px; opacity:0; }} }}
  .agent-active .agent-ring {{ animation: ring-expand 1s ease-out infinite; }}

  /* Beam animations */
  @keyframes beam-draw {{ from {{ stroke-dashoffset:200; opacity:0; }} to {{ stroke-dashoffset:0; opacity:0.55; }} }}
  .beam-animate {{ stroke-dasharray: 6 4; animation: beam-draw 0.4s ease-out forwards; }}

  @keyframes burst-expand {{ 0% {{ r:8px; opacity:0.9; stroke-width:3px; }} 100% {{ r:28px; opacity:0; stroke-width:1px; }} }}
  .burst-animate {{ animation: burst-expand 0.6s ease-out infinite; }}

  .robot-thinking .core-spin {{ animation-duration: 1.2s !important; }}
  .robot-thinking .antenna-pulse {{ animation: antenna-blink 0.6s ease-in-out infinite !important; }}

  /* Sequential reveal */
  @keyframes slide-in-right {{
    from {{ opacity: 0; transform: translateX(22px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
  }}

  #particles {{ position: absolute; top: 0; left: 0; width: 100%; height: 560px; pointer-events: none; }}
</style>
</head>
<body>
<div class="canvas-wrap" id="canvas-wrap">
  <canvas id="particles"></canvas>
  <div class="orch-label">Orchestrator</div>
  <div class="agents-label">Specialists</div>
  <div class="divider-line"></div>

  <svg class="main-svg" viewBox="0 0 780 560" xmlns="http://www.w3.org/2000/svg">
    <g id="beams-layer">{beams_svg}</g>
    <g id="agents-layer">{agents_svg}</g>
    <g id="robot-layer">{robot_svg}</g>
  </svg>

  <div class="info-bar">
    <div class="info-badge">
      <span class="label">Step</span>
      <span class="value">{step}</span>
    </div>
    <div class="info-badge">
      <span class="label">Phase</span>
      <span class="value">{phase}</span>
    </div>
    <div class="info-badge">
      <span class="label">Mode</span>
      <span class="value" style="color:{mode_color};">{mode}</span>
    </div>
    <div class="info-badge">
      <span class="label">Reward</span>
      <span class="value">{reward_html}</span>
    </div>
    <div class="task-text" title="{task_short}">{task_short}</div>
  </div>
</div>

<script>
const STATE = {state_json};

const robotLayer = document.getElementById('robot-layer');
const armRight   = document.getElementById('arm-right');

if (STATE.robot_state === 'thinking' || STATE.robot_state === 'delegating') {{
  robotLayer.classList.add('robot-thinking');
}}
if (STATE.robot_state === 'delegating' && armRight) {{
  armRight.classList.remove('arm-idle');
  armRight.classList.add('arm-delegating');
}}

// Sequential reveal: agents appear one-by-one with staggered delays
if (STATE.mode === 'SEQUENTIAL' && !STATE.done && STATE.called.length > 0) {{
  STATE.called.forEach(function(agentId, idx) {{
    var el = document.getElementById('agent-' + agentId);
    if (!el) return;
    el.style.opacity = '0';
    (function(element, delay) {{
      setTimeout(function() {{
        element.style.transition = 'opacity 0.5s ease';
        element.style.opacity = '1';
      }}, delay);
    }})(el, 250 + idx * 650);
  }});
}}

function spawnParticles(x, y, color) {{
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  const particles = [];
  for (let i = 0; i < 18; i++) {{
    const angle = (Math.PI * 2 * i) / 18;
    const speed = 1.5 + Math.random() * 2.5;
    particles.push({{ x, y, vx: Math.cos(angle)*speed, vy: Math.sin(angle)*speed, life: 1.0, r: 2+Math.random()*2, color }});
  }}
  function animate() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let alive = false;
    particles.forEach(p => {{
      if (p.life <= 0) return;
      p.x += p.vx; p.y += p.vy; p.vx *= 0.92; p.vy *= 0.92; p.life -= 0.025; alive = true;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = color + Math.floor(p.life*255).toString(16).padStart(2,'0');
      ctx.fill();
    }});
    if (alive) requestAnimationFrame(animate);
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
  }}
  animate();
}}

if (STATE.active) {{
  const activeEl = document.getElementById('agent-' + STATE.active);
  if (activeEl) {{
    const wrap  = document.getElementById('canvas-wrap');
    const wRect = wrap.getBoundingClientRect();
    const ct    = activeEl.getCTM();
    if (ct) {{
      const scaleX = wRect.width  / 780;
      const scaleY = wRect.height / 560;
      const tx = ct.e * scaleX;
      const ty = ct.f * scaleY;
      const rect = activeEl.querySelector('rect');
      const agentColor = rect ? rect.getAttribute('stroke') : '#00d4ff';
      setTimeout(() => spawnParticles(tx, ty, agentColor), 300);
    }}
  }}
}}

(function breathe() {{
  const robot = document.getElementById('robot');
  if (!robot) return;
  let t = 0;
  function frame() {{
    t += 0.02;
    const dy = Math.sin(t) * 2.5;
    robot.setAttribute('transform', `translate(160, ${{280 + dy}})`);
    requestAnimationFrame(frame);
  }}
  frame();
}})();
</script>
</body>
</html>"""


# ── State assembler ───────────────────────────────────────────────────────────

def _build_html(state: dict) -> str:
    called      = state.get("called", [])
    active      = state.get("active", "")
    edges       = state.get("edges", [])
    task        = state.get("task", "")
    step        = state.get("step", 0)
    mode        = state.get("mode", "SEQUENTIAL")
    done        = state.get("done", False)
    reward      = state.get("reward", None)
    phase       = state.get("phase", 1)
    spawned_ids = set(state.get("spawned", []))

    # Show only agents that were actually called (+ active if mid-step)
    all_agents = list(called)
    if active and active not in all_agents:
        all_agents.append(active)

    # Nothing delegated yet — robot is idle/thinking, no agent cards needed
    positions = _agent_positions(all_agents) if all_agents else {}

    def agent_status(aid):
        if aid == active: return "active"
        if aid in called: return "done"
        return "idle"

    agents_svg = "\n".join(
        _agent_card_svg(
            aid, *positions[aid],
            agent_status(aid),
            _agent_color(aid, spawned_ids),
            is_spawned=(aid in spawned_ids),
        )
        for aid in all_agents
    )
    beams_svg = _beam_svg(edges, positions)
    robot_svg = _robot_svg()

    robot_state = (
        "delegating" if active else
        "done"       if done   else
        "thinking"   if step > 0 else
        "idle"
    )

    task_short = (task[:72] + "…") if len(task) > 72 else task

    if reward is not None:
        sign         = "+" if reward >= 0 else ""
        reward_color = "#10b981" if reward >= 0 else "#ef4444"
        reward_html  = f'<span style="color:{reward_color};font-weight:700;">{sign}{reward:.3f}</span>'
    else:
        reward_html  = '<span style="color:#334155;">—</span>'

    mode_color = {
        "SEQUENTIAL":     "#00d4ff",
        "PARALLEL":       "#7c3aed",
        "FAN_OUT_REDUCE": "#f59e0b",
        "ITERATIVE":      "#10b981",
        "STOP":           "#ef4444",
    }.get(mode, "#64748b")

    state_json = json.dumps({
        "robot_state": robot_state,
        "active":      active,
        "called":      called,
        "step":        step,
        "done":        done,
        "mode":        mode,
        "spawned":     list(spawned_ids),
    })

    return _html_template(
        agents_svg  = agents_svg,
        beams_svg   = beams_svg,
        robot_svg   = robot_svg,
        state_json  = state_json,
        task_short  = task_short,
        reward_html = reward_html,
        step        = step,
        phase       = phase,
        mode        = mode,
        mode_color  = mode_color,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def render_orchestrator(state: dict, height: int = 600) -> None:
    """
    Render the animated robot orchestrator widget in a Streamlit page.

    state keys:
      called   — list of specialist IDs called so far this episode
      active   — specialist being called right now (or "")
      edges    — list of [caller_id, callee_id] pairs
      task     — task description string
      step     — current step number
      mode     — delegation mode name (e.g. "SEQUENTIAL")
      done     — whether the episode is finished
      reward   — cumulative reward float (or None)
      phase    — curriculum phase int
      spawned  — list of auto-spawned specialist IDs (shown in gold)
    """
    import streamlit.components.v1 as components
    components.html(_build_html(state), height=height, scrolling=False)
