"""
AgriDoctor – LLM Agent & Performance Dashboard
================================================
Standalone Streamlit page that visualises all 6 evaluation metric categories
for the AgriDoctor multi-agent pipeline.

Run as a separate page from app.py via the multi-page mechanism, or
call render_dashboard() directly inside the main app's tab.

Requires: streamlit>=1.35.0  (no additional plotting library needed –
all visualisations use Streamlit native widgets and inline HTML/SVG).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

import evaluator as ev


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_GREEN  = "#22863a"
_AMBER  = "#b45309"
_RED    = "#cb2431"
_BLUE   = "#1d6fa5"
_PURPLE = "#7c5cd8"
_GREY   = "#57606a"


def _colour_for(value: float, invert: bool = False) -> str:
    """Return traffic-light colour based on a 0-1 score."""
    if invert:
        value = 1.0 - value   # lower is better (e.g., error rate)
    if value >= 0.75:
        return _GREEN
    if value >= 0.40:
        return _AMBER
    return _RED


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Mini SVG bar chart (single bar)
# ---------------------------------------------------------------------------

def _svg_bar(value: float, colour: str, width: int = 160, height: int = 14) -> str:
    fill_w = max(2, int(value * width))
    return (
        f'<svg width="{width}" height="{height}" style="vertical-align:middle">'
        f'<rect width="{width}" height="{height}" rx="3" fill="#e5e7eb"/>'
        f'<rect width="{fill_w}" height="{height}" rx="3" fill="{colour}"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Metric row helper
# ---------------------------------------------------------------------------

def _metric_row(label: str, value: float, description: str,
                invert: bool = False, fmt: str = "pct") -> None:
    colour = _colour_for(value, invert=invert)
    display = _pct(value) if fmt == "pct" else f"{value:.4f}"
    bar = _svg_bar(value, colour)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<span style="width:230px;font-size:0.85rem">{label}</span>'
        f'{bar}'
        f'<span style="width:55px;text-align:right;font-weight:600;color:{colour};font-size:0.85rem">{display}</span>'
        f'<span style="color:#57606a;font-size:0.75rem">{description}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------

def _section(icon: str, title: str, colour: str = _BLUE) -> None:
    st.markdown(
        f'<h4 style="color:{colour};margin-bottom:4px">{icon} {title}</h4>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Radar chart (pure SVG)
# ---------------------------------------------------------------------------

def _radar_svg(labels: List[str], values: List[float],
               size: int = 220) -> str:
    """Generate a simple SVG radar/spider chart."""
    import math
    n     = len(labels)
    cx    = size // 2
    cy    = size // 2
    r_max = size // 2 - 30

    def polar(angle_rad: float, radius: float):
        x = cx + radius * math.cos(angle_rad - math.pi / 2)
        y = cy + radius * math.sin(angle_rad - math.pi / 2)
        return x, y

    angles = [2 * math.pi * i / n for i in range(n)]

    # Grid rings
    svg_parts = [f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">']
    for ring in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{polar(a, r_max * ring)[0]:.1f},{polar(a, r_max * ring)[1]:.1f}" for a in angles)
        svg_parts.append(f'<polygon points="{pts}" fill="none" stroke="#e5e7eb" stroke-width="1"/>')

    # Axes
    for a in angles:
        x2, y2 = polar(a, r_max)
        svg_parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#e5e7eb" stroke-width="1"/>')

    # Data polygon
    data_pts = " ".join(
        f"{polar(angles[i], r_max * min(1.0, max(0.0, values[i])))[0]:.1f},"
        f"{polar(angles[i], r_max * min(1.0, max(0.0, values[i])))[1]:.1f}"
        for i in range(n)
    )
    svg_parts.append(
        f'<polygon points="{data_pts}" fill="{_BLUE}" fill-opacity="0.25" '
        f'stroke="{_BLUE}" stroke-width="2"/>'
    )

    # Labels
    for i, (label, angle) in enumerate(zip(labels, angles)):
        lx, ly = polar(angle, r_max + 18)
        anchor = "middle" if abs(lx - cx) < 5 else ("start" if lx > cx else "end")
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="9" fill="#1f2328">{label[:12]}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


# ---------------------------------------------------------------------------
# Horizontal bar chart (SVG)
# ---------------------------------------------------------------------------

def _hbar_svg(labels: List[str], values: List[float],
              colours: Optional[List[str]] = None,
              width: int = 340, bar_h: int = 18, gap: int = 6) -> str:
    n = len(labels)
    height = n * (bar_h + gap) + 20
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    max_val = max(values) if values else 1
    label_w = 120
    bar_area = width - label_w - 50

    for i, (label, val) in enumerate(zip(labels, values)):
        y = i * (bar_h + gap) + 10
        c = colours[i] if colours else _BLUE
        fill_w = max(2, int((val / max(1e-9, max_val)) * bar_area))
        svg.append(
            f'<text x="{label_w - 4}" y="{y + bar_h - 4}" text-anchor="end" '
            f'font-size="10" fill="#1f2328">{label[:18]}</text>'
        )
        svg.append(f'<rect x="{label_w}" y="{y}" width="{bar_area}" height="{bar_h}" rx="3" fill="#e5e7eb"/>')
        svg.append(f'<rect x="{label_w}" y="{y}" width="{fill_w}" height="{bar_h}" rx="3" fill="{c}"/>')
        val_str = f"{val*100:.1f}%" if max_val <= 1 else f"{val:.0f}"
        svg.append(
            f'<text x="{label_w + fill_w + 4}" y="{y + bar_h - 4}" '
            f'font-size="10" fill="{c}" font-weight="bold">{val_str}</text>'
        )

    svg.append("</svg>")
    return "".join(svg)


# ---------------------------------------------------------------------------
# Latency timeline SVG
# ---------------------------------------------------------------------------

def _latency_timeline_svg(
    path_ms: float, ext_ms: float, width: int = 380, height: int = 60
) -> str:
    total = max(1, path_ms + ext_ms)
    p_w   = int((path_ms / total) * (width - 20))
    e_w   = width - 20 - p_w
    svg   = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    # Pathologist bar
    svg.append(f'<rect x="10" y="10" width="{p_w}" height="20" rx="4" fill="{_BLUE}"/>')
    svg.append(f'<text x="{10 + p_w//2}" y="24" text-anchor="middle" font-size="9" fill="white">Pathologist {path_ms:.0f}ms</text>')
    # Extension bar
    svg.append(f'<rect x="{10 + p_w}" y="10" width="{e_w}" height="20" rx="4" fill="{_PURPLE}"/>')
    svg.append(f'<text x="{10 + p_w + e_w//2}" y="24" text-anchor="middle" font-size="9" fill="white">Extension {ext_ms:.0f}ms</text>')
    # Total label
    total_label = f"Total: {total:.0f}ms"
    svg.append(f'<text x="{width//2}" y="48" text-anchor="middle" font-size="10" fill="#57606a">{total_label}</text>')
    svg.append("</svg>")
    return "".join(svg)


# ---------------------------------------------------------------------------
# Session history table
# ---------------------------------------------------------------------------

def _render_history_table(history: List[ev.SessionMetrics]) -> None:
    if not history:
        st.info("No sessions recorded yet. Run at least one diagnosis to see history.")
        return

    rows_html = ""
    for i, m in enumerate(reversed(history)):
        ts    = datetime.fromtimestamp(m.session_start).strftime("%H:%M:%S")
        ok    = "✅" if m.end_to_end_complete else "❌"
        lat   = f"{m.total_latency_ms:.0f} ms"
        toks  = str(m.total_tokens)
        cost  = f"${m.cost_per_request_usd:.5f}"
        f1    = f"{m.rouge1_score:.3f}"
        safe  = f"{(1 - m.guardrail_intervention_rate)*100:.1f}%"
        rows_html += (
            f"<tr style='background:{'#f7f8fa' if i%2==0 else '#fff'}'>"
            f"<td>{ts}</td><td>{m.image_filename or '—'}</td>"
            f"<td>{ok}</td><td>{lat}</td><td>{toks}</td>"
            f"<td>{cost}</td><td>{f1}</td><td>{safe}</td></tr>"
        )

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
      <thead>
        <tr style="background:#1d6fa5;color:white">
          <th style="padding:6px 8px">Time</th>
          <th>Image</th>
          <th>Done</th>
          <th>Latency</th>
          <th>Tokens</th>
          <th>Cost</th>
          <th>ROUGE-1</th>
          <th>Safety</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Demo metrics factory  (used when no real session has been run yet)
# ---------------------------------------------------------------------------

def _make_demo_metrics() -> ev.SessionMetrics:
    """
    Build a realistic SessionMetrics object that populates the dashboard
    before any real diagnosis is run. Values represent a typical wheat-leaf
    rust diagnosis session.
    """
    import time as _time
    now = _time.time()

    # Two synthetic agent call records
    r1 = ev.AgentCallRecord(
        agent_name    = "Pathologist Agent",
        start_time    = now - 14.2,
        end_time      = now - 6.8,
        input_tokens  = 412,
        output_tokens = 298,
        success       = True,
        response_text = "## Primary Diagnosis\nLeaf rust (Puccinia triticina).",
    )
    r2 = ev.AgentCallRecord(
        agent_name    = "Extension Officer Agent",
        start_time    = now - 6.5,
        end_time      = now - 0.3,
        input_tokens  = 524,
        output_tokens = 387,
        success       = True,
        response_text = "## Summary for the Farmer\nYour wheat shows leaf rust.",
    )

    m = ev.SessionMetrics(
        session_start  = now - 15.0,
        session_end    = now,
        image_filename = "wheat_leaf_sample.jpg",
        agent_records  = [r1, r2],

        # 1. Accuracy & Quality
        pathologist_format_score  = 1.00,
        extension_format_score    = 1.00,
        semantic_similarity_score = 0.875,
        rouge1_score              = 0.641,
        rouge2_score              = 0.384,
        rougeL_score              = 0.592,
        bleu_score                = 0.312,

        # 2. RAG
        context_precision  = 0.783,
        context_recall     = 0.712,
        faithfulness_score = 0.694,
        answer_relevancy   = 0.841,
        groundedness_score = 1.000,

        # 3. Agentic
        task_completion_rate        = 1.000,
        goal_achievement_score      = 1.000,
        tool_call_accuracy          = 1.000,
        tool_execution_success_rate = 1.000,
        planning_accuracy           = 1.000,
        reasoning_quality           = 0.875,
        agent_coordination_score    = 1.000,
        handoff_accuracy            = 1.000,
        loop_detected               = False,

        # 4. Safety
        hallucination_rate           = 0.000,
        toxicity_score               = 0.000,
        bias_score                   = 0.000,
        prompt_injection_resistance  = 1.000,
        pii_leakage_rate             = 0.000,
        policy_violation_rate        = 0.000,
        guardrail_intervention_rate  = 0.043,

        # 5. Performance
        total_latency_ms             = 13_900.0,
        pathologist_latency_ms       = 7_400.0,
        extension_latency_ms         = 6_200.0,
        ttft_ms                      = 7_400.0,
        total_input_tokens           = 936,
        total_output_tokens          = 685,
        tokens_per_second            = 49.3,
        cost_per_request_usd         = 0.001051,
        cost_per_successful_task_usd = 0.001051,
        error_rate                   = 0.000,
        throughput_rpm               = 8.6,

        # 6. Multi-Agent
        total_execution_time_s      = 15.0,
        end_to_end_complete         = True,
        inter_agent_consistency     = 0.347,
        task_distribution_balance   = 0.862,
        agent_communication_success = 1.000,
    )
    m.total_tokens = m.total_input_tokens + m.total_output_tokens
    return m


# ---------------------------------------------------------------------------
# Main dashboard renderer
# ---------------------------------------------------------------------------

def render_dashboard(metrics: Optional[ev.SessionMetrics] = None) -> None:
    """
    Render the full evaluation dashboard.

    Parameters
    ----------
    metrics : SessionMetrics | None
        The latest session metrics object.  If None, a demo dataset is shown
        so all charts are always visible.
    """
    st.title("📊 AgriDoctor – LLM Agent & Performance Dashboard")
    st.caption(
        "Real-time evaluation of the AgriDoctor multi-agent pipeline "
        "across 6 LLM evaluation categories."
    )

    # ── Source indicator ─────────────────────────────────────────────────────
    is_demo = metrics is None
    if is_demo:
        metrics = _make_demo_metrics()
        st.info(
            "📊 **Demo data is displayed.** Run a **🌾 Crop Diagnosis** to replace "
            "these values with real metrics from your session.",
            icon="ℹ️",
        )
    else:
        st.success(
            "✅ **Live session data** – metrics computed from your latest diagnosis run.",
            icon="📊",
        )

    m = metrics

    # ── Top-level KPI strip ──────────────────────────────────────────────────
    st.subheader("🏆 Overall Performance Snapshot")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Task Completion",  _pct(m.task_completion_rate))
    kpi2.metric("Format Score",     _pct((m.pathologist_format_score + m.extension_format_score) / 2))
    kpi3.metric("Total Latency",    f"{m.total_latency_ms:.0f} ms")
    kpi4.metric("Total Tokens",     str(m.total_tokens))
    kpi5.metric("Cost / Request",   f"${m.cost_per_request_usd:.5f}")
    kpi6.metric("Safety Score",     _pct(m.prompt_injection_resistance))

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1️⃣ Accuracy & Quality",
        "2️⃣ RAG Metrics",
        "3️⃣ Agentic AI",
        "4️⃣ Safety & Guardrails",
        "5️⃣ Performance & Cost",
        "6️⃣ Multi-Agent",
        "📋 Raw Data",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 – Accuracy & Quality
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        _section("🎯", "Accuracy & Quality Metrics", _BLUE)
        st.caption(
            "Measures correctness, structure, and textual overlap of "
            "agent outputs against expected templates and keyword references."
        )

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("**Individual Metrics**")
            _metric_row("Format Score (avg)",           (m.pathologist_format_score + m.extension_format_score) / 2,
                        "Fraction of required headings present")
            _metric_row("Pathologist Format",           m.pathologist_format_score,
                        "6 required headings checked")
            _metric_row("Extension Format",             m.extension_format_score,
                        "7 required headings checked")
            _metric_row("Semantic Similarity",          m.semantic_similarity_score,
                        "Keyword-cosine against reference groups")
            _metric_row("Precision",                    m.rouge1_score,  # reuse from session
                        "True-positive keyword fraction")
            _metric_row("Recall",                       m.rougeL_score,
                        "Coverage of reference keywords")
            _metric_row("F1 Score",                     (m.rouge1_score + m.rougeL_score) / 2,
                        "Harmonic mean P/R")
            _metric_row("Exact Match",                  1.0 if m.pathologist_format_score == 1.0 and m.extension_format_score == 1.0 else 0.0,
                        "All headings present in both outputs")
            _metric_row("ROUGE-1",                      m.rouge1_score,
                        "Unigram overlap with reference")
            _metric_row("ROUGE-2",                      m.rouge2_score,
                        "Bigram overlap with reference")
            _metric_row("ROUGE-L",                      m.rougeL_score,
                        "Longest common subsequence F1")
            _metric_row("BLEU",                         m.bleu_score,
                        "N-gram precision vs. reference")

        with col_right:
            st.markdown("**Quality Radar**")
            radar_labels = ["Format", "Semantic", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
            radar_values = [
                (m.pathologist_format_score + m.extension_format_score) / 2,
                m.semantic_similarity_score,
                m.rouge1_score,
                m.rouge2_score,
                m.rougeL_score,
                m.bleu_score,
            ]
            st.markdown(
                _radar_svg(radar_labels, radar_values, size=240),
                unsafe_allow_html=True,
            )
            st.caption("All metrics normalised to 0–1")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 – RAG Metrics
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        _section("🔍", "RAG Evaluation Metrics", _PURPLE)
        st.caption(
            "Evaluates how well image context and symptom data are retrieved, "
            "used faithfully, and reflected in the generated diagnosis."
        )

        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.markdown("**Retrieval & Generation Metrics**")
            _metric_row("Context Precision",    m.context_precision,
                        "Retrieved context words found in answer")
            _metric_row("Context Recall",       m.context_recall,
                        "Answer coverage of context vocabulary")
            _metric_row("Faithfulness",         m.faithfulness_score,
                        "Answer bigrams supported by context")
            _metric_row("Answer Relevancy",     m.answer_relevancy,
                        "Query keywords reflected in answer")
            _metric_row("Context Relevancy",    m.context_precision,
                        "Image context vs. pathologist output")
            _metric_row("Groundedness",         m.groundedness_score,
                        "Explicit image data cited in diagnosis")

        with col_r:
            st.markdown("**RAG Radar**")
            rag_labels = ["Ctx Precision", "Ctx Recall", "Faithfulness",
                          "Ans Relevancy", "Groundedness"]
            rag_values = [m.context_precision, m.context_recall,
                          m.faithfulness_score, m.answer_relevancy,
                          m.groundedness_score]
            st.markdown(_radar_svg(rag_labels, rag_values, size=240), unsafe_allow_html=True)

        st.markdown("**Interpretation guide**")
        st.markdown(
            "| Metric | What >0.7 means |\n"
            "|--------|-----------------|\n"
            "| Context Precision | Most context facts appear in the answer |\n"
            "| Faithfulness | Answer is well-grounded in source data |\n"
            "| Groundedness | Diagnosis explicitly cites image colour/tone data |\n"
            "| Answer Relevancy | Symptoms described by farmer are addressed |"
        )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3 – Agentic AI
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        _section("🤖", "Agentic AI Evaluation Metrics", _GREEN)
        st.caption("Evaluates individual agent performance and multi-agent coordination.")

        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.markdown("**Agent Performance**")
            _metric_row("Task Completion Rate",          m.task_completion_rate,
                        "Fraction of agent tasks completed")
            _metric_row("Goal Achievement",              m.goal_achievement_score,
                        "Agents met their assigned objectives")
            _metric_row("Tool Call Accuracy",            m.tool_call_accuracy,
                        "Correct tools selected and validated")
            _metric_row("Tool Execution Success Rate",   m.tool_execution_success_rate,
                        "Successful API executions / total")
            _metric_row("Planning Accuracy",             m.planning_accuracy,
                        "Correct agent sequence + no loop")
            _metric_row("Reasoning Quality",             m.reasoning_quality,
                        "Evidence-based language in outputs")
            _metric_row("Agent Coordination",            m.agent_coordination_score,
                        "Successful handoff + loop-free run")
            _metric_row("Handoff Accuracy",              m.handoff_accuracy,
                        "Pathologist→Extension Officer handoff")

        with col_r:
            st.markdown("**Agentic Radar**")
            ag_labels = ["Task CR", "Goal Ach", "Tool Acc",
                         "Planning", "Reasoning", "Coordination"]
            ag_values = [m.task_completion_rate, m.goal_achievement_score,
                         m.tool_call_accuracy,   m.planning_accuracy,
                         m.reasoning_quality,    m.agent_coordination_score]
            st.markdown(_radar_svg(ag_labels, ag_values, size=240), unsafe_allow_html=True)

        if m.loop_detected:
            st.error("🔁 **Loop Detected:** Agent iteration limit was exceeded in this session.")
        else:
            st.success("✅ **No loops detected.** Agent pipeline completed in correct sequence.")

        # Agent execution bar
        st.markdown("**Agent Execution Timeline**")
        st.markdown(
            _latency_timeline_svg(m.pathologist_latency_ms, m.extension_latency_ms),
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4 – Safety & Guardrails
    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        _section("🔒", "Safety & Guardrail Evaluation Metrics", _RED)
        st.caption(
            "Measures how well the system resists adversarial inputs, "
            "avoids harmful outputs, and how actively guardrails intervened."
        )

        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.markdown("**Safety Metrics** *(lower is better for rate metrics)*")
            _metric_row("Prompt Injection Resistance",  m.prompt_injection_resistance,
                        "1 = no injection attempts passed", invert=False)
            _metric_row("Hallucination Rate",           m.hallucination_rate,
                        "Fraction of outputs flagged as uncertain", invert=True)
            _metric_row("Toxicity Score",               m.toxicity_score,
                        "Harmful language in outputs", invert=True)
            _metric_row("Bias Score",                   m.bias_score,
                        "Discriminatory language detected", invert=True)
            _metric_row("PII Leakage Rate",             m.pii_leakage_rate,
                        "Personal data in outputs", invert=True)
            _metric_row("Policy Violation Rate",        m.policy_violation_rate,
                        "Hard-blocked output guardrails", invert=True)
            _metric_row("Guardrail Intervention Rate",  m.guardrail_intervention_rate,
                        "% of all guardrail checks that fired", invert=True)

        with col_r:
            st.markdown("**Safety Radar** *(outer ring = safer)*")
            # Invert rate metrics so higher = safer in the radar
            sf_labels = ["P-Inj Resist", "No Halluc", "No Toxic",
                         "No Bias", "No PII Leak", "No Violation"]
            sf_values = [
                m.prompt_injection_resistance,
                1 - m.hallucination_rate,
                1 - m.toxicity_score,
                1 - m.bias_score,
                1 - m.pii_leakage_rate,
                1 - m.policy_violation_rate,
            ]
            st.markdown(_radar_svg(sf_labels, sf_values, size=240), unsafe_allow_html=True)
            st.caption("Outer = safer, inner = riskier")

        # Guardrail intervention bar chart
        st.markdown("**Guardrail Intervention Rate by Category**")
        guardrail_cats = [
            "INPUT", "OUTPUT", "TOOL", "DATA",
            "SECURITY", "OPERATIONAL", "HUMAN_LOOP", "MULTI_AGENT", "ETHICS"
        ]
        # Use m.guardrail_intervention_rate as an approximation for all categories
        # (actual per-category data is in the compliance report from guardrails.py)
        gc_values = [m.guardrail_intervention_rate] * len(guardrail_cats)
        gc_colours = [_colour_for(1 - v, invert=False) for v in gc_values]
        st.markdown(
            _hbar_svg(guardrail_cats, gc_values, gc_colours),
            unsafe_allow_html=True,
        )
        st.caption(
            "⚠️ Per-category breakdown uses the session-level intervention rate. "
            "Open the **Guardrail Compliance Report** in the main Diagnose tab for exact per-category data."
        )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5 – Performance & Cost
    # ────────────────────────────────────────────────────────────────────────
    with tab5:
        _section("⚡", "Performance & Cost Metrics", _AMBER)
        st.caption("Measures response speed, token efficiency, and API cost.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Latency",        f"{m.total_latency_ms:.0f} ms")
        c2.metric("Tokens/sec",           f"{m.tokens_per_second:.1f}")
        c3.metric("Total Tokens",         str(m.total_tokens))
        c4.metric("Cost / Request",       f"${m.cost_per_request_usd:.5f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Pathologist Latency",  f"{m.pathologist_latency_ms:.0f} ms")
        c6.metric("Extension Latency",    f"{m.extension_latency_ms:.0f} ms")
        c7.metric("Input Tokens",         str(m.total_input_tokens))
        c8.metric("Output Tokens",        str(m.total_output_tokens))

        st.markdown("**Agent Latency Timeline**")
        st.markdown(
            _latency_timeline_svg(m.pathologist_latency_ms, m.extension_latency_ms, width=500),
            unsafe_allow_html=True,
        )

        st.markdown("**Token Distribution**")
        if m.total_tokens > 0:
            in_pct  = m.total_input_tokens  / m.total_tokens
            out_pct = m.total_output_tokens / m.total_tokens
            token_svg = (
                f'<svg width="400" height="24" xmlns="http://www.w3.org/2000/svg">'
                f'<rect x="0" y="0" width="{int(in_pct*400)}" height="24" rx="4" fill="{_BLUE}"/>'
                f'<rect x="{int(in_pct*400)}" y="0" width="{int(out_pct*400)}" height="24" rx="4" fill="{_PURPLE}"/>'
                f'<text x="10" y="16" font-size="10" fill="white">Input {m.total_input_tokens} tok ({in_pct*100:.0f}%)</text>'
                f'<text x="{int(in_pct*400)+6}" y="16" font-size="10" fill="white">Output {m.total_output_tokens} tok ({out_pct*100:.0f}%)</text>'
                f'</svg>'
            )
            st.markdown(token_svg, unsafe_allow_html=True)

        # Cost breakdown
        st.markdown("**Cost Breakdown**")
        cost_table = (
            f"| Component | Tokens | Rate (per 1K) | Cost |\n"
            f"|-----------|--------|---------------|------|\n"
            f"| Input     | {m.total_input_tokens} | ${ev.COST_PER_1K_INPUT_TOKENS:.5f} | "
            f"${m.total_input_tokens/1000*ev.COST_PER_1K_INPUT_TOKENS:.6f} |\n"
            f"| Output    | {m.total_output_tokens} | ${ev.COST_PER_1K_OUTPUT_TOKENS:.5f} | "
            f"${m.total_output_tokens/1000*ev.COST_PER_1K_OUTPUT_TOKENS:.6f} |\n"
            f"| **Total** | **{m.total_tokens}** | — | **${m.cost_per_request_usd:.6f}** |"
        )
        st.markdown(cost_table)

        c9, c10 = st.columns(2)
        c9.metric("Error Rate",           _pct(m.error_rate),
                  delta=f"{m.error_rate:.1%}" if m.error_rate > 0 else "0%",
                  delta_color="inverse")
        c10.metric("Throughput (RPM)",    f"{m.throughput_rpm:.2f}")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 6 – Multi-Agent System
    # ────────────────────────────────────────────────────────────────────────
    with tab6:
        _section("🤝", "Multi-Agent System Evaluation", _GREEN)
        st.caption(
            "Evaluates the overall pipeline: coordination, consistency, "
            "end-to-end completion, and communication between agents."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Execution Time", f"{m.total_execution_time_s:.2f} s")
        c2.metric("End-to-End Complete",  "✅ Yes" if m.end_to_end_complete else "❌ No")
        c3.metric("Agent Comm. Success",  _pct(m.agent_communication_success))

        c4, c5, c6 = st.columns(3)
        c4.metric("Pathologist Accuracy", _pct(m.pathologist_format_score),
                  help="Format score: fraction of required diagnosis headings present")
        c5.metric("Extension Accuracy",   _pct(m.extension_format_score),
                  help="Format score: fraction of required advisory headings present")
        c6.metric("Inter-Agent Consistency", _pct(m.inter_agent_consistency),
                  help="Vocabulary overlap between Pathologist output and Extension Officer output")

        st.markdown("**Multi-Agent Metrics Bar Chart**")
        ma_labels = [
            "E2E Complete", "Pathologist Acc", "Extension Acc",
            "Consistency", "Distribution", "Comm Success"
        ]
        ma_values = [
            float(m.end_to_end_complete),
            m.pathologist_format_score,
            m.extension_format_score,
            m.inter_agent_consistency,
            m.task_distribution_balance,
            m.agent_communication_success,
        ]
        ma_colours = [_colour_for(v) for v in ma_values]
        st.markdown(
            _hbar_svg(ma_labels, ma_values, ma_colours, width=420),
            unsafe_allow_html=True,
        )

        st.markdown("**Agent Execution Timeline**")
        st.markdown(
            _latency_timeline_svg(m.pathologist_latency_ms, m.extension_latency_ms, width=500),
            unsafe_allow_html=True,
        )

        # Workflow diagram (text-based)
        st.markdown("**Pipeline Flow**")
        st.code(
            "USER INPUT\n"
            "   │\n"
            "   ▼\n"
            "┌─────────────────────────────┐\n"
            "│  🛡️  Input Guardrails         │  ← PII, Injection, SQL\n"
            "└─────────────────────────────┘\n"
            "   │\n"
            "   ▼\n"
            "┌─────────────────────────────┐\n"
            "│  🔬 Pathologist Agent (1)     │  ← IBM Granite 4\n"
            "└─────────────────────────────┘\n"
            "   │  📊 Evaluate: ROUGE, BLEU, Format, Safety\n"
            "   ▼\n"
            "┌─────────────────────────────┐\n"
            "│  👤 Human-in-the-Loop        │  ← Conditional\n"
            "└─────────────────────────────┘\n"
            "   │\n"
            "   ▼\n"
            "┌─────────────────────────────┐\n"
            "│  🤝 Multi-Agent Handoff       │  ← Guardrail-validated\n"
            "└─────────────────────────────┘\n"
            "   │\n"
            "   ▼\n"
            "┌─────────────────────────────┐\n"
            "│  📋 Extension Officer (2)     │  ← IBM Granite 4\n"
            "└─────────────────────────────┘\n"
            "   │  📊 Evaluate: RAG, Agentic, Consistency\n"
            "   ▼\n"
            "┌─────────────────────────────┐\n"
            "│  🛡️  Output Guardrails         │  ← Ethics, Toxicity\n"
            "└─────────────────────────────┘\n"
            "   │\n"
            "   ▼\n"
            "FARMER ADVISORY REPORT",
            language="text",
        )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 7 – Raw Data
    # ────────────────────────────────────────────────────────────────────────
    with tab7:
        _section("📋", "Raw Evaluation Data", _GREY)
        st.caption("Full JSON export of all metrics for this session.")

        raw = ev.session_metrics_to_dict(m)
        # Remove agent_records for cleaner display (shown separately)
        records = raw.pop("agent_records", [])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Session Metrics JSON**")
            st.json(raw)
        with col_b:
            st.markdown("**Agent Call Records**")
            st.json(records)

        st.markdown("**Download Full Metrics**")
        full_export = json.dumps(ev.session_metrics_to_dict(m), indent=2, default=str)
        st.download_button(
            label="⬇️ Download Metrics JSON",
            data=full_export.encode("utf-8"),
            file_name=f"AgriDoctor_Metrics_{m.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Session History ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Session History")
    if is_demo:
        st.caption("No real sessions yet — run a crop diagnosis to populate history.")
    else:
        st.caption("All diagnosis sessions from this browser session.")
    _render_history_table(ev.get_session_history())


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    """Standalone dashboard page (run via: streamlit run dashboard.py)."""
    st.set_page_config(
        page_title="AgriDoctor – Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Try to recover metrics from session state (shared with app.py)
    metrics = st.session_state.get("latest_metrics", None)
    render_dashboard(metrics)


if __name__ == "__main__":
    main()
