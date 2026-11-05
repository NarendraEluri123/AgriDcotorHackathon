"""
AgriDoctor – AI-powered Crop Disease Diagnosis & Advisory
==========================================================
Two-agent workflow:
  1. Pathologist Agent  – analyses a crop/leaf image and produces a diagnosis.
  2. Extension Officer Agent – converts the diagnosis into a farmer-friendly
     remedial-action report.

Both agents call the Replicate REST API with the IBM Granite Vision 4.1 4B
model (ibm-granite/granite-vision-4.1-4b).

Guardrails are applied at every stage of the pipeline:
  • Input  → validate_input, validate_image_file, validate_api_key_format,
             check_jailbreak, check_sql_injection
  • Data   → protect_data (PII masking on image metadata)
  • Tool   → validate_tool_call (before every Replicate API call)
  • Oper.  → check_operational_limits (token budget, rate limit, iterations)
  • Output → validate_output, check_ethical_compliance (per-agent response)
  • Multi  → validate_agent_handoff, verify_cross_agent_output
  • Human  → assess_human_review_need, record_human_decision (UI approval)
  • Audit  → get_compliance_report (shown in sidebar expander)
"""

import base64
import io
import json
import os
import textwrap
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

import guardrails as gr
import evaluator as ev
import dashboard as db

# ---------------------------------------------------------------------------
# Load environment variables from .env (silently if file absent)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# Configuration – Replicate + IBM Granite Vision 4.1 4B
# ---------------------------------------------------------------------------
_ENV_API_KEY   = os.getenv("REPLICATE_API_KEY", "").strip()
MODEL_ID       = os.getenv("REPLICATE_MODEL_ID", "ibm-granite/granite-vision-4.1-4b")
LLM_PROVIDER   = "replicate"
GENERATE_URL   = "https://api.replicate.com/v1/models/{model}/predictions"
PROVIDER_LABEL = f"IBM Granite · {MODEL_ID} via Replicate"


# ---------------------------------------------------------------------------
# Core API call  (Replicate only)
# ---------------------------------------------------------------------------

def call_llm(
    api_key: str,
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    agent_name: str = "agent",
    timer: "ev.AgentTimer | None" = None,
) -> str:
    """
    Send a chat-completion request to Replicate (IBM Granite Vision 4.1 4B)
    and return the assistant response text.

    `messages` follows the OpenAI-style format:
        [{"role": "user"|"system", "content": "..."}, ...]

    Guardrails:  Tool execution + Operational limits applied.
    Evaluation:  Token counts written to timer.record when provided.
    """
    # [GUARDRAIL – TOOL] Validate tool call
    guardrail_endpoint = GENERATE_URL.split("{")[0]
    tool_check = gr.validate_tool_call(
        tool_name="replicate_inference",
        method="POST",
        endpoint=guardrail_endpoint,
        payload={"max_tokens": max_tokens},
    )
    if not tool_check:
        raise RuntimeError(f"[Guardrail] Tool call blocked: {tool_check.reason}")

    # [GUARDRAIL – OPERATIONAL] Check limits before API call
    op_check = gr.check_operational_limits(agent_name, max_tokens)
    if not op_check:
        raise RuntimeError(f"[Guardrail] Operational limit: {op_check.reason}")

    # Estimate input tokens for evaluation
    prompt_text = " ".join(
        (m["content"] if isinstance(m["content"], str) else str(m["content"]))
        for m in messages
    )
    estimated_input_tokens = ev.estimate_tokens(prompt_text)

    response_text, input_tokens, output_tokens = _call_replicate(
        api_key, messages, max_tokens, temperature, estimated_input_tokens
    )

    # Write metrics into the timer record
    if timer is not None:
        timer.record.input_tokens  = input_tokens
        timer.record.output_tokens = output_tokens
        timer.record.response_text = response_text
        timer.record.success       = True

    return response_text


def _call_replicate(
    api_key: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    estimated_input_tokens: int,
) -> tuple:
    """
    Call Replicate's REST API for a language model.
    Returns (response_text, input_tokens, output_tokens).

    Replicate flow:
      1. POST /v1/models/{owner}/{name}/predictions  → get prediction ID
      2. Poll GET /v1/predictions/{id}               → wait for 'succeeded'
      3. Concatenate output tokens into final string

    Messages are converted to a single prompt string:
      system prompt prepended, then user/assistant turns interleaved.
    """
    # Build a single text prompt from OpenAI-style messages
    prompt_parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}")
        elif role in ("assistant", "model"):
            prompt_parts.append(f"<|assistant|>\n{content}")
    prompt_parts.append("<|assistant|>")   # signal the model to start responding
    full_prompt = "\n".join(prompt_parts)

    # Resolve the model URL: replace {model} placeholder
    model_url = GENERATE_URL.format(model=MODEL_ID)

    payload = {
        "input": {
            "prompt":     full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Prefer":        "wait",        # ask Replicate to wait up to 60 s synchronously
    }

    resp = requests.post(model_url, headers=headers, json=payload, timeout=120)

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Replicate API error ({resp.status_code}): {resp.text[:400]}")

    data = resp.json()

    # If 'Prefer: wait' was honoured, the prediction is already complete
    status = data.get("status", "")
    if status not in ("succeeded", "processing", "starting"):
        # Poll until done (fallback for older API versions)
        pred_id   = data.get("id", "")
        poll_url  = f"https://api.replicate.com/v1/predictions/{pred_id}"
        for _ in range(60):          # max 60 poll attempts (~120 s)
            import time as _t; _t.sleep(2)
            poll_resp = requests.get(poll_url, headers=headers, timeout=30)
            if poll_resp.status_code == 200:
                data   = poll_resp.json()
                status = data.get("status", "")
                if status == "succeeded":
                    break
                if status in ("failed", "canceled"):
                    raise RuntimeError(f"Replicate prediction {status}: {data.get('error','unknown')}")

    if data.get("status") != "succeeded":
        raise RuntimeError(f"Replicate prediction did not complete: status={data.get('status')}, error={data.get('error','')}")

    # Output is a list of string tokens – join them
    output = data.get("output", [])
    if isinstance(output, list):
        response_text = "".join(str(t) for t in output)
    else:
        response_text = str(output)

    if not response_text.strip():
        raise RuntimeError("Replicate returned an empty response.")

    # Token counts from metrics if available, else estimate
    metrics       = data.get("metrics", {})
    input_tokens  = int(metrics.get("input_token_count",  estimated_input_tokens))
    output_tokens = int(metrics.get("output_token_count", ev.estimate_tokens(response_text)))
    return response_text, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------

def extract_image_context(uploaded_file) -> dict:
    """
    Extract visual metadata from the uploaded image using PIL.
    Returns a dict with filename, dimensions, mode, dominant colours, and
    brightness — used as structured context for the text-based LLM.
    """
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")
    width, height = img.size
    filename = uploaded_file.name

    # Sample a grid of pixels for colour analysis
    img_small = img.resize((50, 50))
    pixels = list(img_small.getdata())

    # Average colour
    avg_r = int(sum(p[0] for p in pixels) / len(pixels))
    avg_g = int(sum(p[1] for p in pixels) / len(pixels))
    avg_b = int(sum(p[2] for p in pixels) / len(pixels))

    # Brightness (0-255)
    brightness = int((avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114))

    # Colour tone description
    if avg_g > avg_r and avg_g > avg_b:
        if avg_g > 120:
            tone = "predominantly green (healthy-looking foliage)"
        else:
            tone = "dark green (possible shade stress or fungal darkening)"
    elif avg_r > avg_g and avg_r > avg_b:
        tone = "reddish/brownish (possible leaf scorch, rust, or blight)"
    elif avg_r > 150 and avg_g > 130 and avg_b < 100:
        tone = "yellowish-green (possible chlorosis or nutrient deficiency)"
    elif avg_r > 160 and avg_g > 140 and avg_b < 80:
        tone = "yellow (likely chlorosis, nitrogen deficiency, or viral infection)"
    elif avg_r < 80 and avg_g < 80 and avg_b < 80:
        tone = "very dark/black (possible sooty mould, necrosis, or severe blight)"
    else:
        tone = f"mixed tones (R:{avg_r} G:{avg_g} B:{avg_b})"

    brightness_desc = (
        "well-lit" if brightness > 120
        else "dim/underexposed (may affect analysis accuracy)"
    )

    return {
        "filename": filename,
        "dimensions": f"{width}x{height} pixels",
        "colour_tone": tone,
        "avg_rgb": f"R:{avg_r} G:{avg_g} B:{avg_b}",
        "brightness": f"{brightness}/255 ({brightness_desc})",
    }


# ---------------------------------------------------------------------------
# Agent 1 – Pathologist Agent
# ---------------------------------------------------------------------------

PATHOLOGIST_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert plant pathologist and crop disease diagnostician with decades
of field and laboratory experience. A farmer has submitted a crop or plant-leaf
image for diagnosis. You will be given:
  1. Image metadata (dimensions, colour analysis, brightness) extracted
     automatically from the photo.
  2. A description of symptoms as observed by the farmer.

Use both sources of information to deliver a rigorous, evidence-based
preliminary diagnosis.

GUIDELINES:
- Base your analysis on the colour tone data, brightness, and farmer-reported
  symptoms. Colour shifts (yellowing, browning, darkening) are key indicators.
- Identify the most likely disease(s) and state the evidence that supports each.
- Provide a confidence level: High / Medium / Low.
- List up to three alternative differential diagnoses when relevant.
- If the information is insufficient for a reliable assessment, say so clearly
  and ask the farmer for more details rather than guessing.
- Always include the disclaimer that this is an AI-generated preliminary
  assessment and is NOT a confirmed laboratory diagnosis.

OUTPUT FORMAT (use exactly these headings):
## Primary Diagnosis
## Observed Symptoms
## Confidence Level
## Possible Causes
## Differential Diagnoses
## Disclaimer
""").strip()


def run_pathologist_agent(
    api_key: str,
    image_context: dict,
    symptom_description: str,
) -> tuple:
    """
    Agent 1: sends image metadata + farmer symptom description to Granite 4
    and returns (diagnosis_text, AgentCallRecord).

    Guardrails applied:
      • Data guardrail    – PII masking on image context
      • Operational       – token + iteration limits
      • Output            – validate_output, check_ethical_compliance
      • Multi-agent       – marks Pathologist as started
    """
    # [GUARDRAIL – DATA] Protect image context before sending to agent
    data_result = gr.protect_data(image_context, role="user")
    safe_context = data_result.sanitised_value if data_result.sanitised_value else image_context

    context_block = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in safe_context.items()
    )

    user_message = (
        "Please analyse the following crop image data and farmer observations, "
        "then provide a detailed disease diagnosis following the specified "
        "output format.\n\n"
        "=== IMAGE ANALYSIS DATA ===\n"
        f"{context_block}\n\n"
        "=== FARMER-REPORTED SYMPTOMS ===\n"
        f"{symptom_description if symptom_description.strip() else 'No additional description provided by farmer.'}\n\n"
        "Based on the colour analysis and symptoms above, provide your diagnosis."
    )

    messages = [
        {"role": "system", "content": PATHOLOGIST_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # [EVAL] Time the API call
    timer = ev.AgentTimer("Pathologist Agent")
    with timer:
        raw_response = call_llm(
            api_key, messages, max_tokens=1200, temperature=0.2,
            agent_name="Pathologist Agent", timer=timer,
        )

    # [GUARDRAIL – OUTPUT] Validate response from Pathologist Agent
    out_check = gr.validate_output(raw_response, agent_name="Pathologist Agent")
    if not out_check:
        timer.record.success = False
        timer.record.error_message = out_check.reason
        raise RuntimeError(f"[Guardrail] Pathologist output blocked: {out_check.reason}")
    response_text = out_check.sanitised_value if out_check.sanitised_value else raw_response

    # [GUARDRAIL – ETHICS] Ethical and compliance check
    ethics_check = gr.check_ethical_compliance(response_text, agent_name="Pathologist Agent")
    if not ethics_check:
        timer.record.success = False
        raise RuntimeError(f"[Guardrail] Ethical check failed for Pathologist: {ethics_check.reason}")
    # Use potentially disclaimer-enhanced text
    final_text = ethics_check.sanitised_value if ethics_check.sanitised_value else response_text
    timer.record.response_text = final_text

    return final_text, timer.record


# ---------------------------------------------------------------------------
# Agent 2 – Extension Officer Agent
# ---------------------------------------------------------------------------

EXTENSION_OFFICER_SYSTEM_PROMPT = textwrap.dedent("""
You are a friendly and knowledgeable agricultural extension officer with
extensive experience advising smallholder and commercial farmers.  A plant
pathologist has provided you with a crop disease diagnosis report.  Your job is
to convert that technical report into a clear, practical, farmer-friendly
advisory in simple English that even a first-time farmer can understand.

GUIDELINES:
- Use plain, non-technical language wherever possible.
- Provide immediate remedial actions the farmer can take right now.
- Include preventive measures to stop future outbreaks.
- Give crop-care and general best-practice recommendations.
- When chemical treatments are appropriate, mention only well-known, widely
  available options; refer to product labels and local regulations for exact
  dosages – DO NOT invent specific dosages.
- Always recommend Personal Protective Equipment (PPE) when chemicals are
  mentioned.
- End with a safety notice and a recommendation to seek professional help if
  symptoms persist or worsen.
- If the pathologist indicated the image was unclear, advise the farmer to
  obtain a clearer image or consult a local expert directly.

OUTPUT FORMAT (use exactly these headings):
## Summary for the Farmer
## Immediate Remedial Actions
## Preventive Measures
## Crop-Care Recommendations
## Treatment Options
## Safety Warnings
## When to Seek Expert Help
""").strip()


def run_extension_officer_agent(api_key: str, diagnosis_report: str) -> tuple:
    """
    Agent 2: receives the Pathologist's report and produces a farmer-friendly
    advisory. Returns (advisory_text, AgentCallRecord).

    Guardrails applied:
      • Multi-agent   – validate handoff from Pathologist, cross-verify input
      • Operational   – token + iteration limits
      • Output        – validate_output, check_ethical_compliance
    """
    # [GUARDRAIL – MULTI-AGENT] Validate handoff: Pathologist → Extension Officer
    handoff_check = gr.validate_agent_handoff(
        from_agent="Pathologist Agent",
        to_agent="Extension Officer Agent",
        output_length=len(diagnosis_report),
    )
    if not handoff_check:
        raise RuntimeError(f"[Guardrail] Agent handoff blocked: {handoff_check.reason}")

    # [GUARDRAIL – MULTI-AGENT] Cross-verify the Pathologist's output before using it
    crosscheck = gr.verify_cross_agent_output("Pathologist Agent", diagnosis_report)
    if not crosscheck:
        raise RuntimeError(f"[Guardrail] Cross-agent verification failed: {crosscheck.reason}")

    messages = [
        {
            "role": "system",
            "content": EXTENSION_OFFICER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "The plant pathologist has provided the following crop disease "
                "diagnosis report. Please convert it into a practical, "
                "farmer-friendly advisory:\n\n"
                f"{diagnosis_report}"
            ),
        },
    ]

    # [EVAL] Time the API call
    timer = ev.AgentTimer("Extension Officer Agent")
    with timer:
        raw_response = call_llm(
            api_key, messages, max_tokens=1400, temperature=0.4,
            agent_name="Extension Officer Agent", timer=timer,
        )

    # [GUARDRAIL – OUTPUT] Validate Extension Officer response
    out_check = gr.validate_output(raw_response, agent_name="Extension Officer Agent")
    if not out_check:
        timer.record.success = False
        timer.record.error_message = out_check.reason
        raise RuntimeError(f"[Guardrail] Extension Officer output blocked: {out_check.reason}")
    response_text = out_check.sanitised_value if out_check.sanitised_value else raw_response

    # [GUARDRAIL – ETHICS] Ethical compliance check
    ethics_check = gr.check_ethical_compliance(response_text, agent_name="Extension Officer Agent")
    if not ethics_check:
        timer.record.success = False
        raise RuntimeError(f"[Guardrail] Ethical check failed for Extension Officer: {ethics_check.reason}")
    final_text = ethics_check.sanitised_value if ethics_check.sanitised_value else response_text
    timer.record.response_text = final_text

    return final_text, timer.record


# ---------------------------------------------------------------------------
# Report Builder
# ---------------------------------------------------------------------------

def build_final_report(
    diagnosis: str,
    advisory: str,
    filename: str,
) -> str:
    """Combine both agent outputs into a single downloadable text report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider = "=" * 70

    report = textwrap.dedent(f"""
        {divider}
        AGRI DOCTOR – CROP DISEASE ADVISORY REPORT
        Generated: {timestamp}
        Image File: {filename}
        {divider}

        SECTION 1 – PATHOLOGIST AGENT: DISEASE DIAGNOSIS
        {divider}
        {diagnosis}

        {divider}
        SECTION 2 – EXTENSION OFFICER AGENT: FARMER ADVISORY
        {divider}
        {advisory}

        {divider}
        IMPORTANT NOTICE
        {divider}
        This report is an AI-generated preliminary assessment produced by the
        AgriDoctor application using IBM Granite Vision 4.1 4B via Replicate.
        It is NOT a confirmed laboratory diagnosis. Farmers are strongly advised
        to consult a certified plant pathologist or local agricultural extension
        officer for confirmation, especially before applying chemical treatments.
        Always follow local agricultural regulations and pesticide label
        instructions. Use appropriate Personal Protective Equipment (PPE) when
        handling agrochemicals.
        {divider}
    """).strip()

    return report


# ---------------------------------------------------------------------------
# Helper – render guardrail badge
# ---------------------------------------------------------------------------

def _render_guardrail_badge(label: str, passed: bool) -> str:
    colour = "#22863a" if passed else "#cb2431"
    icon   = "✅" if passed else "❌"
    return f'<span style="background:{colour};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.75rem;margin:2px">{icon} {label}</span>'


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AgriDoctor – Crop Disease AI",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # API key comes entirely from .env – never shown in the UI
    api_key = _ENV_API_KEY

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo / title
        st.markdown(
            "<h2 style='margin-bottom:0'>🌿 AgriDoctor</h2>"
            "<p style='color:#57606a;margin-top:2px;font-size:0.85rem'>"
            "AI-Powered Crop Disease Diagnosis</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # System status card
        api_ready = bool(api_key and api_key.strip())
        status_colour = "#22863a" if api_ready else "#cb2431"
        status_icon   = "🟢" if api_ready else "🔴"
        status_text   = "Connected" if api_ready else "No API key found"
        st.markdown(
            f"<div style='background:#f7f8fa;border:1px solid #e5e7eb;"
            f"border-radius:8px;padding:10px 14px;font-size:0.83rem'>"
            f"<b>System Status</b><br>"
            f"<span style='color:{status_colour}'>{status_icon} {status_text}</span><br>"
            f"<span style='color:#57606a'>🤖 {PROVIDER_LABEL}</span><br>"
            f"<span style='color:#57606a'>🌐 api.replicate.com</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # How it works
        st.subheader("ℹ️ How It Works")
        st.markdown(
            """
**① Upload** a crop / leaf photo
**② Guardrails** validate your input
**③ 🔬 Pathologist Agent** diagnoses the disease
**④ Multi-agent handoff** verified by guardrails
**⑤ 📋 Extension Officer** writes farmer-friendly advice
**⑥ Output guardrails** applied
**⑦ Download** your advisory report
            """
        )

        st.divider()

        # About the agents
        st.subheader("🤖 Agents")
        st.markdown(
            f"""
**🔬 Agent 1 – Pathologist**
Analyses image colour data and reported symptoms to identify crop diseases.

**📋 Agent 2 – Extension Officer**
Converts the technical diagnosis into plain-language advice any farmer can act on.

*Powered by {PROVIDER_LABEL}*
            """
        )

        st.divider()

        # Guardrail Status Panel
        st.subheader("🛡️ Guardrail Status")
        st.caption("Live results appear here after each diagnosis run.")
        audit_placeholder = st.empty()

        st.divider()
        st.caption(
            "⚠️ AI-generated preliminary assessment only. "
            "Consult a certified expert before applying any treatment."
        )

    # ── Main Page with Tab Navigation ─────────────────────────────────────────
    # Dashboard is the first tab so it is visible immediately on load
    dash_tab, main_tab = st.tabs(["📊 Performance Dashboard", "🌾 Diagnose Crop"])

    with dash_tab:
        db.render_dashboard(st.session_state.get("latest_metrics", None))

    with main_tab:
        _render_diagnose_tab(api_key, audit_placeholder)


def _render_diagnose_tab(api_key: str, audit_placeholder) -> None:
    """Render the main crop diagnosis tab content."""
    st.title("🌾 AgriDoctor – Crop Disease Diagnosis & Advisory")
    st.markdown(
        "Upload a photo of your crop or plant leaf. AgriDoctor's two AI agents "
        "will diagnose the disease and provide a practical management advisory."
    )

    st.info(
        "📌 **How it works:** "
        "① Upload image → ② Guardrails validate input → "
        "③ Pathologist Agent diagnoses the disease → "
        "④ Multi-agent guardrails verified → "
        "⑤ Extension Officer Agent generates farmer-friendly advice → "
        "⑥ Output guardrails applied → ⑦ Download your advisory report.",
        icon="ℹ️",
    )

    st.divider()

    # ── Image Upload ──────────────────────────────────────────────────────────
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.subheader("📷 Upload Crop Image")
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            help="For best results, upload a clear, well-lit close-up of the "
                 "affected plant part (leaf, stem, fruit, etc.).",
        )
        st.subheader("📝 Describe What You See (Optional but Recommended)")
        symptom_description = st.text_area(
            "Describe the visible symptoms on your crop",
            placeholder=(
                "e.g. 'Leaves are turning yellow with brown spots, edges are "
                "drying out, some leaves have white powdery coating, plant "
                "looks wilted in the morning...'"
            ),
            height=130,
            help="Describe any discolouration, spots, lesions, wilting, mould, "
                 "or unusual appearance you can see. The more detail you provide, "
                 "the more accurate the diagnosis will be.",
        )

    with col_preview:
        if uploaded_file is not None:
            st.subheader("🖼️ Uploaded Image Preview")
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            uploaded_file.seek(0)  # Reset file pointer after PIL reads it

    st.divider()

    # ── Diagnose Button ───────────────────────────────────────────────────────
    diagnose_btn = st.button(
        "🔍 Diagnose Crop",
        type="primary",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )

    if uploaded_file is None:
        st.warning("Please upload a crop or plant-leaf image to get started.")

    # ── Run Agents ────────────────────────────────────────────────────────────
    if diagnose_btn:

        # ── Reset all state for fresh run ────────────────────────────────────
        gr.clear_audit_log()
        gr.reset_operational_counters()
        gr.reset_multi_agent_state()
        session_start = time.time()
        agent_records = []
        handoff_ok = True

        # ════════════════════════════════════════════════════════════════════
        # STAGE 0 – Pre-flight Guardrail Checks
        # ════════════════════════════════════════════════════════════════════
        with st.status("🛡️ Running input guardrails…", expanded=False) as status_box:

            # [GUARDRAIL – INPUT] API key format
            if not api_key:
                st.error(
                    "🔐 API key is not configured. Please add `REPLICATE_API_KEY` "
                    "to your `.env` file and restart the application."
                )
                st.stop()

            key_check = gr.validate_api_key_format(api_key)
            if not key_check:
                st.error(f"🔐 {key_check.reason}")
                st.stop()
            st.write("✅ API key format validated")

            # [GUARDRAIL – INPUT] Image file validation
            img_check = gr.validate_image_file(uploaded_file)
            if not img_check:
                st.error(f"📷 Image validation failed: {img_check.reason}")
                st.stop()
            st.write("✅ Image file validated (format, size, magic bytes)")

            # [GUARDRAIL – SECURITY] Jailbreak + SQL injection on symptom text
            jb_check = gr.check_jailbreak(symptom_description)
            if not jb_check:
                st.error(f"🚫 Security check failed: {jb_check.reason}")
                st.stop()
            st.write("✅ Jailbreak detection passed")

            sql_check = gr.check_sql_injection(symptom_description)
            if not sql_check:
                st.error(f"🚫 SQL injection detected: {sql_check.reason}")
                st.stop()
            st.write("✅ SQL injection check passed")

            # [GUARDRAIL – INPUT] Symptom text validation
            input_check = gr.validate_input(symptom_description)
            if not input_check:
                st.error(f"⚠️ Input validation failed: {input_check.reason}")
                st.stop()
            safe_symptoms = input_check.sanitised_value or symptom_description
            st.write("✅ Input validated (injection, PII, content, relevance)")

            status_box.update(label="🛡️ All input guardrails passed", state="complete")

        # Extract image metadata using PIL
        try:
            image_context = extract_image_context(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to process the uploaded image: {exc}")
            st.stop()

        # ════════════════════════════════════════════════════════════════════
        # STAGE 1 – Agent 1: Pathologist
        # ════════════════════════════════════════════════════════════════════
        st.subheader("🔬 Agent 1 – Pathologist Agent")
        st.caption("Analysing your crop image for disease symptoms…")

        diagnosis_result = None
        path_record = None
        with st.spinner("🧬 Pathologist Agent is examining the image…"):
            try:
                diagnosis_result, path_record = run_pathologist_agent(
                    api_key, image_context, safe_symptoms
                )
                agent_records.append(path_record)
            except RuntimeError as exc:
                error_msg = str(exc)
                if "401" in error_msg or "403" in error_msg:
                    st.error(
                        "❌ Authentication failed. Please check your IBM Cloud "
                        "API key and try again."
                    )
                elif "[Guardrail]" in error_msg:
                    st.error(f"🛡️ Guardrail blocked the Pathologist Agent: {error_msg}")
                elif "model" in error_msg.lower():
                    st.error(
                        "❌ The selected model is not available or not supported "
                        f"for this request.\n\nDetails: {error_msg}"
                    )
                else:
                    st.error(f"❌ Pathologist Agent error: {error_msg}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error in Pathologist Agent: {exc}")
                st.stop()

        with st.expander("📄 Pathologist Agent Report", expanded=True):
            st.markdown(diagnosis_result)

        # ════════════════════════════════════════════════════════════════════
        # STAGE 2 – Human-in-the-Loop Review (if required)
        # ════════════════════════════════════════════════════════════════════
        approval_request = gr.assess_human_review_need(diagnosis_result)

        if approval_request.requires_approval:
            st.divider()
            st.subheader("👤 Human-in-the-Loop Review Required")
            st.warning(
                f"⚠️ **{approval_request.risk_level.upper()} RISK** – {approval_request.description}\n\n"
                "A human review is recommended before the farmer acts on this report. "
                "Please confirm you have reviewed the Pathologist's diagnosis above.",
                icon="⚠️",
            )

            col_approve, col_reject = st.columns(2)
            with col_approve:
                human_approved = st.button(
                    "✅ Reviewed & Approved – Continue",
                    type="primary",
                    use_container_width=True,
                    key="human_approve",
                )
            with col_reject:
                human_rejected = st.button(
                    "❌ Reject – Stop Pipeline",
                    use_container_width=True,
                    key="human_reject",
                )

            if human_rejected:
                gr.record_human_decision(approval_request, approved=False)
                st.error(
                    "🛑 Pipeline stopped by human reviewer. "
                    "Please consult a certified plant pathologist directly."
                )
                st.stop()

            if not human_approved:
                st.info(
                    "👆 Please click **Reviewed & Approved** above to proceed to "
                    "the Extension Officer advisory, or click **Reject** to stop."
                )
                st.stop()

            gr.record_human_decision(approval_request, approved=True)
            st.success("✅ Human review completed. Proceeding to Extension Officer Agent.")
        else:
            # Low / medium risk – record automatic pass
            gr.record_human_decision(approval_request, approved=True)

        # ════════════════════════════════════════════════════════════════════
        # STAGE 3 – Agent 2: Extension Officer
        # ════════════════════════════════════════════════════════════════════
        st.subheader("📋 Agent 2 – Extension Officer Agent")
        st.caption(
            "Converting the diagnosis into a farmer-friendly advisory…"
        )

        advisory_result = None
        ext_record = None
        with st.spinner("📝 Extension Officer Agent is preparing your advisory…"):
            try:
                advisory_result, ext_record = run_extension_officer_agent(
                    api_key, diagnosis_result
                )
                agent_records.append(ext_record)
            except RuntimeError as exc:
                error_msg = str(exc)
                if "401" in error_msg or "403" in error_msg:
                    st.error(
                        "❌ Authentication failed. Please check your IBM Cloud "
                        "API key and try again."
                    )
                elif "[Guardrail]" in error_msg:
                    st.error(f"🛡️ Guardrail blocked the Extension Officer Agent: {error_msg}")
                    handoff_ok = False
                else:
                    st.error(f"❌ Extension Officer Agent error: {error_msg}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error in Extension Officer Agent: {exc}")
                st.stop()

        with st.expander("📄 Extension Officer Advisory", expanded=True):
            st.markdown(advisory_result)

        # ════════════════════════════════════════════════════════════════════
        # STAGE 4 – Evaluate Session & Store Metrics
        # ════════════════════════════════════════════════════════════════════
        session_end = time.time()
        guardrail_audit = gr.get_audit_log()

        session_metrics = ev.evaluate_session(
            agent_records        = agent_records,
            pathologist_output   = diagnosis_result or "",
            extension_output     = advisory_result or "",
            image_context        = image_context,
            symptom_description  = safe_symptoms,
            guardrail_audit_log  = guardrail_audit,
            session_start        = session_start,
            session_end          = session_end,
            handoff_successful   = handoff_ok,
            loop_detected        = False,
            image_filename       = uploaded_file.name,
        )
        st.session_state["latest_metrics"] = session_metrics

        # ════════════════════════════════════════════════════════════════════
        # STAGE 5 – Combined Final Report
        # ════════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("📑 Combined Final Report")

        final_report = build_final_report(
            diagnosis_result,
            advisory_result,
            uploaded_file.name,
        )

        st.success("✅ Multi-agent analysis complete! View the **📊 Performance Dashboard** tab for full evaluation metrics.")

        with st.expander("📋 View Full Combined Report", expanded=False):
            st.text(final_report)

        # ── Download Button ───────────────────────────────────────────────────
        st.download_button(
            label="⬇️ Download Advisory Report (.txt)",
            data=final_report.encode("utf-8"),
            file_name=f"AgriDoctor_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # ════════════════════════════════════════════════════════════════════
        # STAGE 6 – Compliance Report (Sidebar + Expander)
        # ════════════════════════════════════════════════════════════════════
        compliance = gr.get_compliance_report()
        cat_summary = compliance.get("by_category", {})

        # Sidebar audit badge update
        badge_html = ""
        category_icons = {
            "INPUT":       "📥",
            "OUTPUT":      "📤",
            "TOOL":        "🔧",
            "DATA":        "💾",
            "SECURITY":    "🔒",
            "OPERATIONAL": "⚙️",
            "HUMAN_LOOP":  "👤",
            "MULTI_AGENT": "🤝",
            "ETHICS":      "⚖️",
        }
        for cat, stats in cat_summary.items():
            icon  = category_icons.get(cat, "🛡️")
            label = f"{icon} {cat}"
            all_passed = stats["failed"] == 0
            badge_html += _render_guardrail_badge(label, all_passed) + " "

        with audit_placeholder:
            st.markdown(badge_html, unsafe_allow_html=True)
            st.caption(
                f"**{compliance['overall_passed']}** checks passed · "
                f"**{compliance['overall_failed']}** checks flagged"
            )

        # Detailed compliance expander in main area
        st.divider()
        with st.expander("🛡️ Guardrail Compliance Report", expanded=False):
            st.subheader("Guardrail Compliance Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Checks", compliance["total_checks"])
            col2.metric("Passed", compliance["overall_passed"])
            col3.metric(
                "Flagged",
                compliance["overall_failed"],
                delta=None if compliance["overall_failed"] == 0 else f"-{compliance['overall_failed']}",
                delta_color="inverse",
            )

            st.markdown("#### Results by Category")
            for cat, stats in cat_summary.items():
                icon = category_icons.get(cat, "🛡️")
                status = "✅ All Passed" if stats["failed"] == 0 else f"⚠️ {stats['failed']} Flagged"
                st.markdown(
                    f"**{icon} {cat}** – {stats['total']} checks · {status}"
                )

            st.markdown("#### Full Audit Trail")
            st.json(compliance["audit_log"])

        # ── Footer Warning ────────────────────────────────────────────────────
        st.divider()
        st.warning(
            "⚠️ **Important Disclaimer:** This report is an AI-generated "
            "preliminary assessment and is NOT a confirmed laboratory diagnosis. "
            "Always consult a certified plant pathologist or local agricultural "
            "extension officer before taking action, especially regarding "
            "chemical treatments. Follow all pesticide label instructions and "
            "local agricultural regulations. Use appropriate Personal Protective "
            "Equipment (PPE) when handling agrochemicals.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
