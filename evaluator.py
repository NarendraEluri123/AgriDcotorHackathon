"""
AgriDoctor – LLM Evaluation Metrics Engine
============================================
Computes and stores evaluation metrics across all 6 categories:

  1. Accuracy & Quality Metrics  (Accuracy, Precision, Recall, F1, EM,
                                   Semantic Similarity, ROUGE-1/2/L, BLEU)
  2. RAG Evaluation Metrics       (Context Precision, Recall, Faithfulness,
                                   Answer Relevancy, Groundedness)
  3. Agentic AI Metrics           (Task Completion Rate, Goal Achievement,
                                   Tool Call Accuracy, Planning/Reasoning,
                                   Agent Coordination, Handoff Accuracy,
                                   Loop Detection)
  4. Safety & Guardrail Metrics   (Hallucination Rate, Toxicity Score,
                                   Bias Score, Prompt Injection Resistance,
                                   PII Leakage Rate, Policy Violation Rate,
                                   Guardrail Intervention Rate)
  5. Performance & Cost Metrics   (Latency, TTFT, Tokens/sec, Input/Output
                                   Tokens, Cost per Request, Throughput,
                                   Error Rate)
  6. Multi-Agent System Metrics   (Total execution time, End-to-end
                                   completion, Individual agent accuracy,
                                   Inter-agent consistency, Task distribution)

All computation is done using the Python standard library only.
No NLTK, spacy, or sentence-transformers are required.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cost constants (Replicate – illustrative USD per 1 K tokens)
# ---------------------------------------------------------------------------
COST_PER_1K_INPUT_TOKENS  = 0.00035   # $ per 1,000 input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.00105   # $ per 1,000 output tokens


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class AgentCallRecord:
    """One complete LLM API call record."""
    agent_name:       str
    start_time:       float          # Unix timestamp before API call
    end_time:         float          # Unix timestamp after API call
    input_tokens:     int  = 0
    output_tokens:    int  = 0
    success:          bool = True
    error_message:    str  = ""
    response_text:    str  = ""
    # Guardrail outcomes for this call
    guardrail_blocked: bool = False
    guardrail_checks:  int  = 0
    guardrail_passed:  int  = 0

    @property
    def latency_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class SessionMetrics:
    """Accumulated metrics for one full diagnosis session (both agents)."""
    session_id:         str   = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    session_start:      float = field(default_factory=time.time)
    session_end:        float = 0.0
    image_filename:     str   = ""
    agent_records:      List[AgentCallRecord] = field(default_factory=list)

    # ── 1. Accuracy & Quality (computed post-hoc) ──────────────────────────
    pathologist_format_score:   float = 0.0   # 0-1: fraction of required headings present
    extension_format_score:     float = 0.0
    semantic_similarity_score:  float = 0.0   # keyword-overlap based cosine proxy
    rouge1_score:               float = 0.0
    rouge2_score:               float = 0.0
    rougeL_score:               float = 0.0
    bleu_score:                 float = 0.0

    # ── 2. RAG Metrics ──────────────────────────────────────────────────────
    context_precision:          float = 0.0
    context_recall:             float = 0.0
    faithfulness_score:         float = 0.0
    answer_relevancy:           float = 0.0
    groundedness_score:         float = 0.0

    # ── 3. Agentic Metrics ──────────────────────────────────────────────────
    task_completion_rate:       float = 0.0   # 0 or 1 per session (both agents done)
    goal_achievement_score:     float = 0.0
    tool_call_accuracy:         float = 0.0
    tool_execution_success_rate: float = 0.0
    planning_accuracy:          float = 0.0
    reasoning_quality:          float = 0.0
    agent_coordination_score:   float = 0.0
    handoff_accuracy:           float = 1.0
    loop_detected:              bool  = False

    # ── 4. Safety Metrics ───────────────────────────────────────────────────
    hallucination_rate:         float = 0.0
    toxicity_score:             float = 0.0
    bias_score:                 float = 0.0
    prompt_injection_resistance: float = 1.0
    pii_leakage_rate:           float = 0.0
    policy_violation_rate:      float = 0.0
    guardrail_intervention_rate: float = 0.0

    # ── 5. Performance Metrics ───────────────────────────────────────────────
    total_latency_ms:           float = 0.0
    pathologist_latency_ms:     float = 0.0
    extension_latency_ms:       float = 0.0
    ttft_ms:                    float = 0.0   # estimated (same as latency for sync calls)
    total_input_tokens:         int   = 0
    total_output_tokens:        int   = 0
    total_tokens:               int   = 0
    tokens_per_second:          float = 0.0
    cost_per_request_usd:       float = 0.0
    cost_per_successful_task_usd: float = 0.0
    error_rate:                 float = 0.0
    throughput_rpm:             float = 0.0   # requests per minute

    # ── 6. Multi-Agent Metrics ────────────────────────────────────────────
    total_execution_time_s:     float = 0.0
    end_to_end_complete:        bool  = False
    inter_agent_consistency:    float = 0.0   # keyword overlap between outputs
    task_distribution_balance:  float = 0.0   # |t1-t2| / (t1+t2)  → 1 = perfectly equal
    agent_communication_success: float = 1.0


# Global session history (in-memory; persisted in Streamlit session_state)
_session_history: List[SessionMetrics] = []


def get_session_history() -> List[SessionMetrics]:
    return list(_session_history)


def clear_session_history() -> None:
    _session_history.clear()


def record_session(m: SessionMetrics) -> None:
    _session_history.append(m)


# ===========================================================================
# Timing context manager
# ===========================================================================

class AgentTimer:
    """Context manager that records start/end time into an AgentCallRecord."""

    def __init__(self, agent_name: str):
        self.record = AgentCallRecord(
            agent_name=agent_name,
            start_time=0.0,
            end_time=0.0,
        )

    def __enter__(self) -> "AgentTimer":
        self.record.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.record.end_time = time.perf_counter()
        if exc_type is not None:
            self.record.success = False
            self.record.error_message = str(exc_val)
        return False   # do not suppress exceptions


# ===========================================================================
# Token estimation (no tiktoken dependency)
# ===========================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate: ~4 chars per token (GPT-family heuristic).
    Acceptable for cost & performance estimation without tiktoken.
    """
    return max(1, len(text) // 4)


# ===========================================================================
# 1. Accuracy & Quality Metrics
# ===========================================================================

_PATHOLOGIST_HEADINGS = [
    "Primary Diagnosis", "Observed Symptoms", "Confidence Level",
    "Possible Causes", "Differential Diagnoses", "Disclaimer",
]

_EXTENSION_HEADINGS = [
    "Summary for the Farmer", "Immediate Remedial Actions",
    "Preventive Measures", "Crop-Care Recommendations",
    "Treatment Options", "Safety Warnings", "When to Seek Expert Help",
]

# Reference keywords per section (used as lightweight ground-truth proxy)
_PATHOLOGIST_REFERENCE_KEYWORDS = {
    "disease_terms": ["disease", "pathogen", "fungal", "bacterial", "viral",
                      "blight", "rust", "mould", "mold", "rot", "spot",
                      "lesion", "chlorosis", "necrosis", "wilt"],
    "evidence_terms": ["colour", "color", "symptom", "indicates", "suggests",
                       "based on", "evidence", "caused by", "due to"],
    "confidence_terms": ["high", "medium", "low", "confidence"],
    "disclaimer_terms": ["preliminary", "not a confirmed", "consult", "certified",
                         "ai-generated", "laboratory"],
}

_EXTENSION_REFERENCE_KEYWORDS = {
    "action_terms": ["remove", "apply", "spray", "isolate", "prune", "water",
                     "treat", "clean", "avoid", "use"],
    "safety_terms": ["ppe", "protective", "gloves", "mask", "safety",
                     "precaution", "warning"],
    "expert_terms": ["consult", "expert", "pathologist", "extension officer",
                     "professional", "laboratory"],
}


def _tokenise(text: str) -> List[str]:
    """Simple word tokeniser (lowercase, alpha-only)."""
    return re.findall(r"[a-z]+", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _rouge_n(hypothesis: str, reference: str, n: int) -> float:
    """ROUGE-N: recall-oriented n-gram overlap."""
    hyp_tokens = _tokenise(hypothesis)
    ref_tokens = _tokenise(reference)
    if not ref_tokens:
        return 0.0
    hyp_ng = Counter(_ngrams(hyp_tokens, n))
    ref_ng = Counter(_ngrams(ref_tokens, n))
    overlap = sum(min(hyp_ng[g], ref_ng[g]) for g in ref_ng)
    return overlap / max(1, sum(ref_ng.values()))


def _rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L: longest common subsequence F1."""
    hyp = _tokenise(hypothesis)
    ref = _tokenise(reference)
    if not hyp or not ref:
        return 0.0
    # DP LCS length
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / max(1, m)
    recall    = lcs / max(1, n)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU with up to max_n n-grams and brevity penalty."""
    hyp_tokens = _tokenise(hypothesis)
    ref_tokens = _tokenise(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    log_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_ngrams(hyp_tokens, n))
        ref_ng = Counter(_ngrams(ref_tokens, n))
        clipped = sum(min(cnt, ref_ng[g]) for g, cnt in hyp_ng.items())
        total   = max(1, sum(hyp_ng.values()))
        prec    = clipped / total if total > 0 else 0
        if prec == 0:
            return 0.0
        log_sum += math.log(prec)

    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / max(1, len(hyp_tokens))
    )
    return bp * math.exp(log_sum / max_n)


def _keyword_cosine(text: str, keyword_groups: Dict[str, List[str]]) -> float:
    """
    Lightweight semantic similarity proxy.
    Measures what fraction of reference keyword groups are represented in the text.
    Returns a value in [0, 1].
    """
    text_lower = text.lower()
    hits = 0
    total = len(keyword_groups)
    for group_keywords in keyword_groups.values():
        if any(kw in text_lower for kw in group_keywords):
            hits += 1
    return hits / max(1, total)


def _format_score(text: str, required_headings: List[str]) -> float:
    """Fraction of required headings present in the response."""
    found = sum(1 for h in required_headings if h in text)
    return found / max(1, len(required_headings))


def compute_accuracy_quality(
    pathologist_output: str,
    extension_output: str,
) -> Dict[str, float]:
    """
    Compute Accuracy & Quality metrics.
    Since we have no labelled ground-truth dataset, we use:
      • Format score  → fraction of required section headings present
      • Semantic similarity → keyword-cosine against reference keyword groups
      • ROUGE-1/2/L & BLEU → self-reference (output vs. expected keywords
        concatenated as a pseudo-reference string)

    Returns a dict of metric_name → score (all 0-1 or 0-100 where noted).
    """
    # Build pseudo-reference strings from keyword groups
    path_ref = " ".join(
        w for grp in _PATHOLOGIST_REFERENCE_KEYWORDS.values() for w in grp
    )
    ext_ref = " ".join(
        w for grp in _EXTENSION_REFERENCE_KEYWORDS.values() for w in grp
    )

    p_fmt  = _format_score(pathologist_output, _PATHOLOGIST_HEADINGS)
    e_fmt  = _format_score(extension_output,   _EXTENSION_HEADINGS)
    fmt    = (p_fmt + e_fmt) / 2

    p_sem  = _keyword_cosine(pathologist_output, _PATHOLOGIST_REFERENCE_KEYWORDS)
    e_sem  = _keyword_cosine(extension_output,   _EXTENSION_REFERENCE_KEYWORDS)
    sem    = (p_sem + e_sem) / 2

    combined = pathologist_output + " " + extension_output
    r1 = _rouge_n(combined, path_ref + " " + ext_ref, 1)
    r2 = _rouge_n(combined, path_ref + " " + ext_ref, 2)
    rl = _rouge_l(combined, path_ref + " " + ext_ref)
    bl = _bleu(combined,    path_ref + " " + ext_ref)

    # Precision / Recall / F1 derived from keyword coverage
    # Precision: of keywords mentioned, how many are from the reference set?
    hyp_words  = set(_tokenise(combined))
    ref_words  = set(_tokenise(path_ref + " " + ext_ref))
    tp = len(hyp_words & ref_words)
    fp = len(hyp_words - ref_words)
    fn = len(ref_words - hyp_words)
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / max(1e-9, precision + recall))

    # Exact Match: 1 if all required headings present, else 0
    em = 1.0 if (
        all(h in pathologist_output for h in _PATHOLOGIST_HEADINGS) and
        all(h in extension_output   for h in _EXTENSION_HEADINGS)
    ) else 0.0

    return {
        "format_score":          round(fmt,       4),
        "semantic_similarity":   round(sem,       4),
        "precision":             round(precision, 4),
        "recall":                round(recall,    4),
        "f1_score":              round(f1,        4),
        "exact_match":           round(em,        4),
        "rouge1":                round(r1,        4),
        "rouge2":                round(r2,        4),
        "rougeL":                round(rl,        4),
        "bleu":                  round(bl,        4),
    }


# ===========================================================================
# 2. RAG Evaluation Metrics
# ===========================================================================

def compute_rag_metrics(
    pathologist_output: str,
    extension_output:   str,
    image_context:      Dict[str, Any],
    symptom_description: str,
) -> Dict[str, float]:
    """
    Compute RAG-style metrics.
    In this pipeline the "retrieved context" = image_context + symptom_description.
    The "generated answer" = pathologist_output + extension_output.

    Metrics are computed via keyword-overlap heuristics since no vector DB exists.
    """
    context_text = " ".join(str(v) for v in image_context.values()) + " " + symptom_description
    answer_text  = pathologist_output + " " + extension_output

    # Context Precision: do retrieved facts appear verbatim in the answer?
    context_words = set(_tokenise(context_text))
    answer_words  = set(_tokenise(answer_text))
    ctx_precision = len(context_words & answer_words) / max(1, len(context_words))

    # Context Recall: does the answer cover the key context terms?
    ctx_recall = len(answer_words & context_words) / max(1, len(context_words))

    # Faithfulness: are claims in the answer supported by the context?
    # Proxy: fraction of answer unique n-grams that have overlap with context
    ans_bigrams = set(_ngrams(_tokenise(answer_text), 2))
    ctx_bigrams = set(_ngrams(_tokenise(context_text), 2))
    faithful = len(ans_bigrams & ctx_bigrams) / max(1, len(ans_bigrams))

    # Answer Relevancy: does the answer address the symptom query?
    query_keywords = _tokenise(symptom_description) if symptom_description.strip() else []
    if query_keywords:
        matched = sum(1 for kw in query_keywords if kw in answer_text.lower())
        relevancy = matched / max(1, len(query_keywords))
    else:
        relevancy = 0.5   # no query → neutral

    # Context Relevancy: is the image context relevant to the answer?
    # Proxy: colour tone / brightness keywords reflected in the pathologist output
    ctx_relevancy = _rouge_n(pathologist_output, context_text, 1)

    # Groundedness: does the answer explicitly cite image data?
    grounding_phrases = [
        "colour", "color", "brightness", "rgb", "tone", "green", "yellow",
        "brown", "red", "dark", "image", "photo", "pixel", "visual",
    ]
    grounded_hits = sum(1 for p in grounding_phrases if p in pathologist_output.lower())
    groundedness  = min(1.0, grounded_hits / 5)   # saturate at 5 matches

    return {
        "context_precision":   round(ctx_precision, 4),
        "context_recall":      round(ctx_recall,    4),
        "faithfulness":        round(faithful,      4),
        "answer_relevancy":    round(relevancy,     4),
        "context_relevancy":   round(ctx_relevancy, 4),
        "groundedness":        round(groundedness,  4),
    }


# ===========================================================================
# 3. Agentic AI Metrics
# ===========================================================================

def compute_agentic_metrics(
    agent_records:       List[AgentCallRecord],
    handoff_successful:  bool,
    loop_detected:       bool,
    guardrail_outcomes:  Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute Agentic AI evaluation metrics from call records and outcomes.
    """
    total_agents   = len(_PATHOLOGIST_HEADINGS)   # expected = 2 agents
    success_count  = sum(1 for r in agent_records if r.success)
    task_cr        = success_count / max(1, len(agent_records))

    # Goal achievement: both agents completed AND output has all required headings
    # (simplified to task_cr if we don't have full outputs)
    goal_ach = task_cr

    # Tool call accuracy: guardrail-validated tool calls / total tool calls
    tool_calls_total   = guardrail_outcomes.get("tool_total", 0)
    tool_calls_passed  = guardrail_outcomes.get("tool_passed", 0)
    tool_call_acc      = tool_calls_passed / max(1, tool_calls_total)

    # Tool execution success rate
    tool_exec_success  = success_count / max(1, len(agent_records))

    # Planning accuracy: did agents run in the correct sequence?
    # Proxy: 1 if handoff successful + no loop
    planning_acc = 1.0 if (handoff_successful and not loop_detected) else 0.5

    # Reasoning quality: average semantic similarity across all agent outputs
    reasoning_scores = []
    for rec in agent_records:
        if rec.success and rec.response_text:
            kw_group = (_PATHOLOGIST_REFERENCE_KEYWORDS
                        if "Pathologist" in rec.agent_name
                        else _EXTENSION_REFERENCE_KEYWORDS)
            reasoning_scores.append(_keyword_cosine(rec.response_text, kw_group))
    reasoning_q = sum(reasoning_scores) / max(1, len(reasoning_scores))

    # Agent coordination: handoff accuracy + no-loop penalty
    agent_coord = (1.0 if handoff_successful else 0.0) * (0.0 if loop_detected else 1.0)
    handoff_acc = 1.0 if handoff_successful else 0.0

    return {
        "task_completion_rate":       round(task_cr,          4),
        "goal_achievement":           round(goal_ach,         4),
        "tool_call_accuracy":         round(tool_call_acc,    4),
        "tool_execution_success_rate": round(tool_exec_success, 4),
        "planning_accuracy":          round(planning_acc,     4),
        "reasoning_quality":          round(reasoning_q,      4),
        "agent_coordination":         round(agent_coord,      4),
        "handoff_accuracy":           round(handoff_acc,      4),
        "loop_detected":              float(loop_detected),
    }


# ===========================================================================
# 4. Safety & Guardrail Metrics
# ===========================================================================

def compute_safety_metrics(
    pathologist_output:  str,
    extension_output:    str,
    guardrail_audit_log: List[Dict[str, Any]],
    input_text:          str = "",
) -> Dict[str, float]:
    """
    Compute Safety & Guardrail evaluation metrics from agent outputs and
    the guardrail audit trail.
    """
    combined = pathologist_output + " " + extension_output

    # Hallucination rate: fraction of outputs that triggered hallucination guardrail
    hallucination_triggers = [
        r for r in guardrail_audit_log
        if r.get("check") == "hallucination_detection" and not r.get("passed")
    ]
    total_output_checks = max(1, sum(
        1 for r in guardrail_audit_log if r.get("category") == "OUTPUT"
    ))
    hallucination_rate = len(hallucination_triggers) / total_output_checks

    # Toxicity score: fraction of toxicity checks that failed
    tox_fails = sum(
        1 for r in guardrail_audit_log
        if r.get("check") == "toxicity_filter" and not r.get("passed")
    )
    toxicity_score = tox_fails / total_output_checks

    # Bias score: fraction of bias checks that failed
    bias_fails = sum(
        1 for r in guardrail_audit_log
        if r.get("check") == "bias_detection" and not r.get("passed")
    )
    total_ethics = max(1, sum(
        1 for r in guardrail_audit_log if r.get("category") == "ETHICS"
    ))
    bias_score = bias_fails / total_ethics

    # Prompt injection resistance: 1 - (injection failures / injection checks)
    inj_checks = [
        r for r in guardrail_audit_log if r.get("check") == "prompt_injection"
    ]
    inj_fails   = sum(1 for r in inj_checks if not r.get("passed"))
    pi_resist   = 1.0 - (inj_fails / max(1, len(inj_checks)))

    # PII leakage rate: any PII detected in OUTPUT category
    pii_leaks = [
        r for r in guardrail_audit_log
        if r.get("check") == "sensitive_data_filter" and not r.get("passed")
    ]
    pii_leak_rate = len(pii_leaks) / total_output_checks

    # Policy violation rate: any output guardrail hard-blocked
    policy_violations = [
        r for r in guardrail_audit_log
        if r.get("category") == "OUTPUT" and not r.get("passed")
    ]
    policy_viol_rate = len(policy_violations) / total_output_checks

    # Guardrail intervention rate: total failed checks / total checks
    total_checks  = max(1, len(guardrail_audit_log))
    total_failed  = sum(1 for r in guardrail_audit_log if not r.get("passed"))
    intervention_rate = total_failed / total_checks

    return {
        "hallucination_rate":           round(hallucination_rate,  4),
        "toxicity_score":               round(toxicity_score,       4),
        "bias_score":                   round(bias_score,           4),
        "prompt_injection_resistance":  round(pi_resist,            4),
        "pii_leakage_rate":             round(pii_leak_rate,        4),
        "policy_violation_rate":        round(policy_viol_rate,     4),
        "guardrail_intervention_rate":  round(intervention_rate,    4),
    }


# ===========================================================================
# 5. Performance & Cost Metrics
# ===========================================================================

def compute_performance_metrics(
    agent_records:    List[AgentCallRecord],
    session_start:    float,
    session_end:      float,
) -> Dict[str, Any]:
    """
    Compute Performance & Cost metrics from timing and token counts.
    """
    if not agent_records:
        return {}

    total_lat_ms  = sum(r.latency_ms  for r in agent_records)
    path_records  = [r for r in agent_records if "Pathologist" in r.agent_name]
    ext_records   = [r for r in agent_records if "Extension"   in r.agent_name]

    path_lat_ms   = path_records[0].latency_ms if path_records else 0.0
    ext_lat_ms    = ext_records[0].latency_ms  if ext_records  else 0.0

    total_in_tok  = sum(r.input_tokens  for r in agent_records)
    total_out_tok = sum(r.output_tokens for r in agent_records)
    total_tok     = total_in_tok + total_out_tok

    # Tokens per second (based on total latency)
    total_lat_s   = total_lat_ms / 1000.0
    tps           = total_out_tok / max(1e-6, total_lat_s)

    # Cost
    cost_in  = (total_in_tok  / 1000) * COST_PER_1K_INPUT_TOKENS
    cost_out = (total_out_tok / 1000) * COST_PER_1K_OUTPUT_TOKENS
    cost_req = cost_in + cost_out

    success_count = sum(1 for r in agent_records if r.success)
    cost_task = cost_req / max(1, success_count)

    error_count = sum(1 for r in agent_records if not r.success)
    error_rate  = error_count / max(1, len(agent_records))

    session_duration_s = session_end - session_start if session_end > 0 else total_lat_s
    throughput_rpm = (len(agent_records) / max(1e-6, session_duration_s)) * 60

    return {
        "total_latency_ms":            round(total_lat_ms, 2),
        "pathologist_latency_ms":      round(path_lat_ms,  2),
        "extension_latency_ms":        round(ext_lat_ms,   2),
        "ttft_ms":                     round(path_lat_ms,  2),   # sync: TTFT ≈ agent latency
        "total_input_tokens":          total_in_tok,
        "total_output_tokens":         total_out_tok,
        "total_tokens":                total_tok,
        "tokens_per_second":           round(tps,           2),
        "cost_per_request_usd":        round(cost_req,      6),
        "cost_per_successful_task_usd": round(cost_task,    6),
        "error_rate":                  round(error_rate,    4),
        "throughput_rpm":              round(throughput_rpm, 2),
    }


# ===========================================================================
# 6. Multi-Agent System Metrics
# ===========================================================================

def compute_multi_agent_metrics(
    agent_records:       List[AgentCallRecord],
    pathologist_output:  str,
    extension_output:    str,
    session_start:       float,
    session_end:         float,
    handoff_successful:  bool,
) -> Dict[str, float]:
    """
    Compute Multi-Agent System metrics.
    """
    session_dur = session_end - session_start if session_end > 0 else 0.0
    e2e_complete = float(
        len(agent_records) == 2 and all(r.success for r in agent_records)
    )

    # Inter-agent consistency: overlap between pathologist output and extension output
    # Measures whether the extension officer actually used the diagnosis
    p_words = set(_tokenise(pathologist_output))
    e_words = set(_tokenise(extension_output))
    if p_words | e_words:
        consistency = len(p_words & e_words) / len(p_words | e_words)
    else:
        consistency = 0.0

    # Task distribution balance: equal latency split → higher balance score
    path_records = [r for r in agent_records if "Pathologist" in r.agent_name]
    ext_records  = [r for r in agent_records if "Extension"   in r.agent_name]
    t1 = path_records[0].latency_ms if path_records else 0
    t2 = ext_records[0].latency_ms  if ext_records  else 0
    if t1 + t2 > 0:
        balance = 1.0 - abs(t1 - t2) / (t1 + t2)
    else:
        balance = 0.0

    comm_success = 1.0 if handoff_successful else 0.0

    # Individual agent accuracy (format score proxy)
    p_acc = _format_score(pathologist_output, _PATHOLOGIST_HEADINGS)
    e_acc = _format_score(extension_output,   _EXTENSION_HEADINGS)

    return {
        "total_execution_time_s":       round(session_dur,    2),
        "end_to_end_complete":          round(e2e_complete,   4),
        "pathologist_accuracy":         round(p_acc,          4),
        "extension_accuracy":           round(e_acc,          4),
        "inter_agent_consistency":      round(consistency,    4),
        "task_distribution_balance":    round(balance,        4),
        "agent_communication_success":  round(comm_success,   4),
    }


# ===========================================================================
# Master evaluation function
# ===========================================================================

def evaluate_session(
    agent_records:       List[AgentCallRecord],
    pathologist_output:  str,
    extension_output:    str,
    image_context:       Dict[str, Any],
    symptom_description: str,
    guardrail_audit_log: List[Dict[str, Any]],
    session_start:       float,
    session_end:         float,
    handoff_successful:  bool  = True,
    loop_detected:       bool  = False,
    image_filename:      str   = "",
) -> SessionMetrics:
    """
    Run the full evaluation suite for one session.
    Returns a populated SessionMetrics object.
    """
    m = SessionMetrics(
        session_start   = session_start,
        session_end     = session_end,
        image_filename  = image_filename,
    )
    m.agent_records = agent_records

    # ── 1. Accuracy & Quality ───────────────────────────────────────────────
    aq = compute_accuracy_quality(pathologist_output, extension_output)
    m.pathologist_format_score  = _format_score(pathologist_output, _PATHOLOGIST_HEADINGS)
    m.extension_format_score    = _format_score(extension_output,   _EXTENSION_HEADINGS)
    m.semantic_similarity_score = aq["semantic_similarity"]
    m.rouge1_score              = aq["rouge1"]
    m.rouge2_score              = aq["rouge2"]
    m.rougeL_score              = aq["rougeL"]
    m.bleu_score                = aq["bleu"]

    # ── 2. RAG ──────────────────────────────────────────────────────────────
    rag = compute_rag_metrics(
        pathologist_output, extension_output, image_context, symptom_description
    )
    m.context_precision   = rag["context_precision"]
    m.context_recall      = rag["context_recall"]
    m.faithfulness_score  = rag["faithfulness"]
    m.answer_relevancy    = rag["answer_relevancy"]
    m.groundedness_score  = rag["groundedness"]

    # ── 3. Agentic ──────────────────────────────────────────────────────────
    tool_total  = sum(1 for r in guardrail_audit_log if r.get("category") == "TOOL")
    tool_passed = sum(1 for r in guardrail_audit_log
                      if r.get("category") == "TOOL" and r.get("passed"))
    ag = compute_agentic_metrics(
        agent_records, handoff_successful, loop_detected,
        {"tool_total": tool_total, "tool_passed": tool_passed}
    )
    m.task_completion_rate       = ag["task_completion_rate"]
    m.goal_achievement_score     = ag["goal_achievement"]
    m.tool_call_accuracy         = ag["tool_call_accuracy"]
    m.tool_execution_success_rate = ag["tool_execution_success_rate"]
    m.planning_accuracy          = ag["planning_accuracy"]
    m.reasoning_quality          = ag["reasoning_quality"]
    m.agent_coordination_score   = ag["agent_coordination"]
    m.handoff_accuracy           = ag["handoff_accuracy"]
    m.loop_detected              = bool(ag["loop_detected"])

    # ── 4. Safety ───────────────────────────────────────────────────────────
    sf = compute_safety_metrics(
        pathologist_output, extension_output, guardrail_audit_log, symptom_description
    )
    m.hallucination_rate          = sf["hallucination_rate"]
    m.toxicity_score              = sf["toxicity_score"]
    m.bias_score                  = sf["bias_score"]
    m.prompt_injection_resistance = sf["prompt_injection_resistance"]
    m.pii_leakage_rate            = sf["pii_leakage_rate"]
    m.policy_violation_rate       = sf["policy_violation_rate"]
    m.guardrail_intervention_rate = sf["guardrail_intervention_rate"]

    # ── 5. Performance ──────────────────────────────────────────────────────
    perf = compute_performance_metrics(agent_records, session_start, session_end)
    m.total_latency_ms            = perf.get("total_latency_ms",             0.0)
    m.pathologist_latency_ms      = perf.get("pathologist_latency_ms",       0.0)
    m.extension_latency_ms        = perf.get("extension_latency_ms",         0.0)
    m.ttft_ms                     = perf.get("ttft_ms",                      0.0)
    m.total_input_tokens          = perf.get("total_input_tokens",           0)
    m.total_output_tokens         = perf.get("total_output_tokens",          0)
    m.total_tokens                = perf.get("total_tokens",                 0)
    m.tokens_per_second           = perf.get("tokens_per_second",            0.0)
    m.cost_per_request_usd        = perf.get("cost_per_request_usd",         0.0)
    m.cost_per_successful_task_usd = perf.get("cost_per_successful_task_usd", 0.0)
    m.error_rate                  = perf.get("error_rate",                   0.0)
    m.throughput_rpm              = perf.get("throughput_rpm",               0.0)

    # ── 6. Multi-Agent ──────────────────────────────────────────────────────
    ma = compute_multi_agent_metrics(
        agent_records, pathologist_output, extension_output,
        session_start, session_end, handoff_successful
    )
    m.total_execution_time_s     = ma["total_execution_time_s"]
    m.end_to_end_complete        = bool(ma["end_to_end_complete"])
    m.inter_agent_consistency    = ma["inter_agent_consistency"]
    m.task_distribution_balance  = ma["task_distribution_balance"]
    m.agent_communication_success = ma["agent_communication_success"]

    record_session(m)
    return m


def session_metrics_to_dict(m: SessionMetrics) -> Dict[str, Any]:
    """Convert a SessionMetrics to a flat JSON-serialisable dict."""
    d = asdict(m)
    # agent_records contain floats/strings – already serialisable
    return d
