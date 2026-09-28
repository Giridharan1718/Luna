"""
LunarAI — AI-powered multi-modal lunar image correspondence & registration platform.

Generated shell — edit `app_build.py` (shell) or `app/lib/` (design system + pages),
then run `python app_build.py`.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from lunarai_lib import appdata, live
from lunarai_lib.config import load_config

from app.lib import components as ui
from app.lib import pages as page_registry
from app.lib import theme
from app.lib.pages import judge

cfg = load_config()

st.set_page_config(
    page_title="LunarAI · ISRO SIH 2026 · PS26166",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# the nine destinations (kept in sync with app/lib/pages/__init__.py)
NAV = [
    "🌕 Mission Dashboard",
    "📤 Dataset Manager",
    "🧠 LunaDNA Retrieval",
    "📍 Image Registration",
    "📊 Validation Analytics",
    "📑 Reports & Exports",
    "🛰 Explainability",
    "⚙ System Health",
    "ℹ About LunarAI",
]

theme.inject()

# --------------------------------------------------------------------------- state
st.session_state.setdefault("nav", NAV[0])
st.session_state.setdefault("judge_mode", False)
st.session_state.setdefault("theme", "deep space")
st.session_state.setdefault("role", "ISRO Scientist")

ROLES = {
    "ISRO Scientist": ("Mission Dashboard", False),
    "Planetary Researcher": ("LunaDNA Retrieval", False),
    "Remote Sensing Expert": ("Image Registration", False),
    "SIH Judge": ("Mission Dashboard", True),
    "AI Engineer": ("System Health", False),
}

# a quick-link button (from any page) asks for a destination before the nav renders
_request = st.session_state.pop("nav_request", None)
if _request in NAV:
    st.session_state["nav"] = _request

_h = appdata.headline(cfg)


def _notifications() -> list[tuple[str, str]]:
    """Honest advisories surfaced in the notification centre."""
    notes: list[tuple[str, str]] = []
    try:
        import cv2

        if not hasattr(cv2, "AKAZE_create"):
            notes.append(("warning", "AKAZE comparison unavailable — this OpenCV build "
                                     "ships without contrib modules."))
    except Exception:
        pass
    kaguya = Path(cfg.DATA_ROOT) / "kaya"
    if kaguya.is_dir():
        tiles = list(kaguya.glob("*.img"))
        if len(tiles) > 1:
            notes.append(("info", f"{len(tiles)} KAGUYA SELENE TC tiles ingested; the "
                                  "second is truncated and treated as such."))
    if not (cfg.OUTPUTS_ROOT / "sih_complete" / "robustness_validation.csv").is_file():
        notes.append(("warning", "Robustness evidence missing — run "
                                 "`scripts/run_sih_complete.py`."))
    stale = _h.get("epoch")
    if stale is not None:
        notes.append(("accent", f"LunaDNA checkpoint on disk is epoch {stale}; live pages "
                                "use these exact weights."))
    if not notes:
        notes.append(("success", "All subsystems nominal."))
    return notes


# ---------------------------------------------------------------- top mission bar
judge_mode = bool(st.session_state["judge_mode"])
ui.topbar([
    ("mission", "OPERATIONAL"),
    ("dataset", f"{_h.get('images', '—')} imgs · {_h.get('patches', '—')} patches"),
    ("model", f"LunaDNA · epoch {_h.get('epoch', '—')}"),
    ("experiment", f"{_h.get('runs', '—')} pipeline runs"),
    ("role", str(st.session_state["role"])),
], judge_mode=judge_mode)

notes = _notifications()
bar = st.columns([1.1, 1.1, 1.4, 1.2, 3])
with bar[0]:
    if st.button("🏆 Exit judge mode" if judge_mode else "🏆 Judge mode",
                 key="judge_toggle",
                 type="secondary" if judge_mode else "primary"):
        st.session_state["judge_mode"] = not judge_mode
        st.rerun()
with bar[1]:
    label = "☀ Console" if st.session_state["theme"] == "deep space" else "🌌 Deep space"
    if st.button(label, key="theme_toggle"):
        st.session_state["theme"] = ("console" if st.session_state["theme"] == "deep space"
                                     else "deep space")
        st.rerun()
with bar[2]:
    role = bar[2].selectbox("View as", list(ROLES), key="role_select",
                            label_visibility="collapsed")
    if role != st.session_state["role"]:
        st.session_state["role"] = role
        dest, wants_judge = ROLES[role]
        st.session_state["nav"] = next((n for n in NAV if n.endswith(dest)), NAV[0])
        st.session_state["judge_mode"] = wants_judge
        st.rerun()
with bar[3]:
    with st.popover(f"🔔 Notifications ({len(notes)})"):
        for tone, text in notes:
            st.markdown(ui.chip(tone, tone) + " " + text, unsafe_allow_html=True)
with bar[4]:
    st.caption("LunarAI mission console · every figure is generated from the artifacts "
               "on this machine.")

if st.session_state["theme"] == "console":
    st.markdown(f"<style>{theme.light_override()}</style>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="lai-brand">
          <div class="lai-brand-disc"></div>
          <div class="lai-brand-name">LunarAI</div>
          <div class="lai-brand-tag">Mission Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if judge_mode:
        st.markdown(f"<div class='lai-nav-label'>Presentation</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='lai-card' style='padding:12px 14px'>"
            "<h4 style='margin:0 0 6px 0'>Judge mode active</h4>"
            "<div style='font-size:0.8rem; color:var(--lai-text-2)'>Technical controls are "
            "hidden. Use the top bar to return to the mission console.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='lai-sidebar-foot'><b>Narrative</b><br>"
            "1 Problem · 2 Solution · 3 Pipeline<br>4 Results · 5 Benchmarks<br>"
            "6 Innovation · 7 Live demo<br>8 Impact · 9 Robustness · 10 Future scope</div>",
            unsafe_allow_html=True,
        )
        page = NAV[0]
    else:
        st.markdown(f"<div class='lai-nav-label'>Navigate</div>", unsafe_allow_html=True)
        page = st.radio("Navigation", NAV, key="nav", label_visibility="collapsed")
        current = page_registry.get(next((p.key for p in page_registry.PAGES
                                          if f"{p.icon} {p.label}" == page), "dashboard"))
        st.markdown(
            f"<div class='lai-sidebar-foot'><b>{current.group} layer</b><br>"
            f"{current.label}</div>",
            unsafe_allow_html=True,
        )

    si = live.session_info(cfg)
    st.markdown(
        f"""
        <div class="lai-sidebar-foot">
          <b>Session</b><br>
          device&nbsp;&nbsp;: {si.get('device', 'cpu')}<br>
          matcher: {si.get('matcher') or 'loads on first live run'}<br>
          vectors: {si.get('ntotal') if si.get('ntotal') is not None else '—'}<br>
          images&nbsp;: {_h.get('images', '—')}<br>
          patches: {_h.get('patches', '—')}<br>
          <span style="opacity:0.7">v3.0 · Streamlit + PyTorch + FAISS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------- router
if judge_mode:
    judge.render(cfg)
else:
    key = next((p.key for p in page_registry.PAGES if f"{p.icon} {p.label}" == page),
               "dashboard")
    page_registry.get(key).render(cfg)
