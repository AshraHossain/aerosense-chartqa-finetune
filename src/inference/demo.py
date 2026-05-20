"""
src/inference/demo.py

Streamlit demo for AeroSense ChartQA fine-tuned model.
Run: streamlit run src/inference/demo.py
"""

from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AeroSense ChartQA",
    page_icon="✈️",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️ AeroSense ChartQA")
st.markdown(
    "**Fine-tuned aviation chart Q&A** — Qwen2.5-3B + LoRA/QLoRA  \n"
    "Ask any question about aeronautical charts, approach procedures, airspace, or FAA regulations."
)
st.divider()

# ── Example questions ─────────────────────────────────────────────────────────
with st.expander("💡 Example questions"):
    examples = [
        "What does a dashed magenta circle on a VFR sectional chart indicate?",
        "What is the difference between decision altitude and minimum descent altitude?",
        "What are the pilot certification requirements to fly in Class B airspace?",
        "How does GPS RAIM work and why does it matter for IFR operations?",
        "What are the fuel requirements for an IFR flight?",
        "What does a blue airport symbol mean on a VFR sectional chart?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question"] = ex

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g. What does a magenta airport symbol mean on a sectional chart?",
)

col1, col2 = st.columns([1, 3])
with col1:
    max_tokens = st.slider("Max tokens", 128, 1024, 512, 64)

submit = st.button("Ask AeroSense", type="primary", use_container_width=True)

# ── Inference ─────────────────────────────────────────────────────────────────
if submit and question.strip():
    with st.spinner("Generating answer..."):
        try:
            response = requests.post(
                f"{API_BASE}/infer",
                json={"question": question, "max_tokens": max_tokens},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("safety_flagged"):
                st.warning("⚠️ Safety notice: This query was flagged. Answer includes safety context.")

            st.markdown("### Answer")
            st.markdown(data["answer"])

            st.divider()
            col_a, col_b = st.columns(2)
            col_a.metric("Latency", f"{data['latency_ms']:.0f} ms")
            col_b.metric("Model", data["model"])

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to inference API. "
                "Make sure the FastAPI server is running: `uvicorn src.inference.api:app --reload`"
            )
        except Exception as e:
            st.error(f"Error: {e}")

elif submit:
    st.warning("Please enter a question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AeroSense ChartQA · Fine-tuned on synthetic aviation Q&A from FAA AIM, TERPS, and chart specifications  \n"
    "Built by [Ashrafuzzaman M. Hossain](https://linkedin.com/in/ashrafmhossain) · "
    "[GitHub](https://github.com/AshraHossain/aerosense-chartqa-finetune) · "
    "[Hugging Face](https://huggingface.co/AshraHossain)"
)
