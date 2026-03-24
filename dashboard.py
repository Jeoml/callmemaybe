"""
CallSense Dashboard — conversational chat UI with trace inspection and prompt management.
The assistant is the orchestrator + agents working together. Each message goes through
the full graph pipeline and shows routing/eval metadata inline.
Run: streamlit run dashboard.py
"""

import json
import os
import re
import uuid

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("CALLSENSE_API_URL", "http://localhost:8000")
TRACES_PATH = os.getenv("TRACES_PATH", "./traces")

st.set_page_config(page_title="CallSense", layout="wide")


# ── Helpers ─────────────────────────────────────────────────────────────────

def agent_color(agent: str) -> str:
    return {"rag": "#2dd4bf", "web": "#a78bfa", "both": "#3b82f6", "escalate": "#9ca3af"}.get(agent, "#6b7280")


def conf_color(conf: str) -> str:
    return {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(conf, "#6b7280")


def badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.75em;font-weight:600">{text}</span>'


def _score_color(score) -> str:
    """Return a color based on score value (1-5)."""
    if score is None:
        return "#6b7280"
    if score >= 4:
        return "#22c55e"
    if score >= 3:
        return "#f59e0b"
    return "#ef4444"


def _score_bar(score, max_score=5) -> str:
    """Return an inline HTML score bar."""
    if score is None:
        return '<span style="color:#6b7280;font-size:0.8em">N/A</span>'
    pct = (score / max_score) * 100
    color = _score_color(score)
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="flex:1;background:#333;border-radius:4px;height:8px;overflow:hidden">'
        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px"></div></div>'
        f'<span style="color:{color};font-weight:600;font-size:0.9em;min-width:32px">{score}/{max_score}</span>'
        f'</div>'
    )


EVAL_DIMENSIONS = [
    ("routing_accuracy", "Routing Accuracy", "Was the query routed to the right agent?"),
    ("faithfulness", "Faithfulness", "Is the response grounded in the retrieved context?"),
    ("resolution_confidence", "Resolution Confidence", "Does the response fully answer the query?"),
]


def render_eval_bar(meta: dict) -> None:
    """Render agent/confidence badges, expandable eval scores with reasoning, and trace ID."""
    agent = meta.get("agent_used", "?")
    confidence = meta.get("routing_confidence", "?")
    scores = meta.get("eval_scores", {})
    trace_id = meta.get("trace_id", "")

    # Agent + confidence badges
    st.markdown(
        f'{badge(agent.upper(), agent_color(agent))} '
        f'{badge(confidence.upper(), conf_color(confidence))}',
        unsafe_allow_html=True,
    )

    # Expandable eval scores
    score_summary = []
    for key, label, _ in EVAL_DIMENSIONS:
        s = scores.get(key, {})
        sv = s.get("score")
        if sv is not None:
            score_summary.append(f"{label}: {sv}/5")
    summary_text = " · ".join(score_summary) if score_summary else "No scores"

    with st.expander(f"Eval Scores — {summary_text}"):
        for key, label, description in EVAL_DIMENSIONS:
            s = scores.get(key, {})
            sv = s.get("score")
            reason = s.get("reason", "")

            st.markdown(
                f'<div style="margin-bottom:4px">'
                f'<span style="font-weight:600;font-size:0.9em">{label}</span> '
                f'<span style="color:#888;font-size:0.75em">— {description}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_score_bar(sv), unsafe_allow_html=True)
            if reason:
                st.caption(reason)
            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

        st.markdown(
            f'<span style="color:#555;font-size:0.75em">Trace ID:</span>',
            unsafe_allow_html=True,
        )
        st.code(trace_id, language=None)


_COT_SPLIT_RE = re.compile(r'(?m)^(\d+\.\s)')
_ROUTING_SUMMARY_RE = re.compile(r'(Routing summary:\s*.+)$', re.DOTALL)


def parse_cot_sections(cot_text: str) -> list[dict]:
    """
    Parse chain-of-thought reasoning into numbered sections.
    Each section has a 'header' (the numbered question) and 'content' (the reasoning).
    Also captures a trailing 'Routing summary' if present.
    """
    if not cot_text or not cot_text.strip():
        return []

    sections = []
    parts = _COT_SPLIT_RE.split(cot_text.strip())

    # parts[0] is any text before the first numbered section
    preamble = parts[0].strip()
    if preamble:
        sections.append({"header": "Preamble", "content": preamble})

    # Remaining parts come in pairs: (number_prefix, content)
    i = 1
    while i < len(parts) - 1:
        num_prefix = parts[i]  # e.g. "1. "
        body = parts[i + 1]
        # First line of body is the header text, rest is the reasoning
        lines = body.strip().split("\n", 1)
        header_text = f"{num_prefix}{lines[0].strip()}"
        content = lines[1].strip() if len(lines) > 1 else ""
        sections.append({"header": header_text, "content": content})
        i += 2

    # Check for a trailing routing summary after all numbered sections
    if sections:
        last_content = sections[-1]["content"]
        summary_match = _ROUTING_SUMMARY_RE.search(last_content)
        if summary_match:
            # Pull the summary out of the last section's content
            sections[-1]["content"] = last_content[:summary_match.start()].strip()
            sections.append({"header": "Routing Summary", "content": summary_match.group(1).replace("Routing summary:", "").strip()})

    return sections


def render_cot(cot_text: str):
    """Render CoT as section headers (always visible) with expandable content."""
    sections = parse_cot_sections(cot_text)
    if not sections:
        st.caption("No reasoning recorded.")
        return
    for i, section in enumerate(sections):
        with st.expander(section["header"], expanded=False):
            if section["content"]:
                st.markdown(section["content"])
            else:
                st.caption("(no additional detail)")


def load_traces_from_disk() -> list[dict]:
    traces = []
    if not os.path.exists(TRACES_PATH):
        return traces
    for fname in os.listdir(TRACES_PATH):
        if not fname.endswith(".jsonl"):
            continue
        try:
            with open(os.path.join(TRACES_PATH, fname), "r", encoding="utf-8") as f:
                traces.append(json.load(f))
        except Exception:
            pass
    traces.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return traces


# ── Session state init ──────────────────────────────────────────────────────

if "conversations" not in st.session_state:
    # conversations: dict of conv_id -> { "name": str, "messages": list[dict] }
    # each message: { "role": "user"|"assistant", "content": str, "meta": dict|None }
    st.session_state.conversations = {}

if "active_conv" not in st.session_state:
    st.session_state.active_conv = None

if "page" not in st.session_state:
    st.session_state.page = "chat"


# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("CallSense")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Chat", "Traces", "Batch Eval", "Prompt Manager"],
    index=["chat", "traces", "batch_eval", "prompts"].index(st.session_state.page)
    if st.session_state.page in ["chat", "traces", "batch_eval", "prompts"]
    else 0,
    key="nav_radio",
)
st.session_state.page = {"Chat": "chat", "Traces": "traces", "Batch Eval": "batch_eval", "Prompt Manager": "prompts"}[page]

st.sidebar.divider()

# ── Sidebar: Conversations list (shown on Chat page) ───────────────────────

if st.session_state.page == "chat":
    st.sidebar.subheader("Conversations")

    if st.sidebar.button("+ New conversation", use_container_width=True):
        conv_id = str(uuid.uuid4())[:8]
        st.session_state.conversations[conv_id] = {
            "name": f"Chat {len(st.session_state.conversations) + 1}",
            "messages": [],
        }
        st.session_state.active_conv = conv_id
        st.rerun()

    for conv_id, conv in st.session_state.conversations.items():
        # Show conversation name with message count
        msg_count = len([m for m in conv["messages"] if m["role"] == "user"])
        label = f'{conv["name"]} ({msg_count} msgs)'
        is_active = conv_id == st.session_state.active_conv
        if st.sidebar.button(
            f"{'> ' if is_active else ''}{label}",
            key=f"conv_{conv_id}",
            use_container_width=True,
        ):
            st.session_state.active_conv = conv_id
            st.rerun()


# ── Page: Chat ──────────────────────────────────────────────────────────────

if st.session_state.page == "chat":
    st.title("Chat with Assistant")

    conv_id = st.session_state.active_conv
    if not conv_id or conv_id not in st.session_state.conversations:
        st.info("Create a new conversation from the sidebar to start chatting.")
    else:
        conv = st.session_state.conversations[conv_id]

        # Render conversation history
        for i, msg in enumerate(conv["messages"]):
            with st.chat_message(msg["role"]):
                meta = msg.get("meta")

                # For assistant messages: Thinking (collapsed) → Response → Evals
                if meta and msg["role"] == "assistant":
                    cot_text = meta.get("cot_reasoning", "")
                    if cot_text:
                        with st.status("Thought process", state="complete", expanded=False):
                            st.markdown(cot_text)

                st.markdown(msg["content"])

                if meta and msg["role"] == "assistant":
                    render_eval_bar(meta)

        # Chat input
        if user_input := st.chat_input("Ask the assistant..."):
            # Add user message
            conv["messages"].append({"role": "user", "content": user_input, "meta": None})

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Call the streaming API
            with st.chat_message("assistant"):
                answer = ""
                meta = {"trace_id": "", "agent_used": "", "intent": "", "cot_reasoning": "", "routing_confidence": "", "eval_scores": {}}
                cot_buffer = ""
                resp_buffer = ""

                # ── Phase 1: Thinking (Grok-style — spinner, streams, collapses) ──
                thinking_status = st.status("Thinking...", expanded=True)
                cot_placeholder = thinking_status.empty()
                resp_placeholder = st.empty()

                try:
                    resp = requests.post(
                        f"{API_BASE}/call/stream",
                        json={"query": user_input},
                        stream=True,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data: "):
                                continue
                            payload = line[6:]
                            if payload == "[DONE]":
                                break
                            try:
                                data = json.loads(payload)
                            except json.JSONDecodeError:
                                continue

                            evt_type = data.get("type", "")

                            if evt_type == "cot_delta":
                                cot_buffer += data.get("content", "")
                                cot_placeholder.markdown(cot_buffer + "▌")

                            elif evt_type == "response_delta":
                                resp_buffer += data.get("content", "")
                                resp_placeholder.markdown(resp_buffer + "▌")

                            elif evt_type == "error":
                                answer = f"Error: {data.get('message', 'Unknown error')}"
                                thinking_status.update(label="Error", state="error", expanded=False)

                            elif evt_type == "node":
                                node = data.get("node", "")

                                if node == "orchestrator":
                                    cot_placeholder.markdown(cot_buffer)
                                    meta["intent"] = data.get("intent", "")
                                    meta["routing_confidence"] = data.get("routing_confidence", "")
                                    meta["trace_id"] = data.get("trace_id", "")
                                    meta["cot_reasoning"] = cot_buffer
                                    thinking_status.update(label="Thought process", state="complete", expanded=False)

                                elif node == "eval":
                                    meta["eval_scores"] = data.get("eval_scores", {})
                                    meta["trace_id"] = data.get("trace_id", meta["trace_id"])

                                else:
                                    meta["agent_used"] = node

                        # Finalize response
                        answer = resp_buffer.strip() or answer
                        if answer:
                            resp_placeholder.markdown(answer)
                        else:
                            answer = "No response received."
                            resp_placeholder.markdown(answer)
                    else:
                        answer = f"Error: API returned {resp.status_code}"
                        resp_placeholder.markdown(answer)
                        thinking_status.update(label="Error", state="error", expanded=False)
                        meta = None
                except requests.exceptions.ConnectionError:
                    answer = "Could not connect to API. Is the server running? (`uvicorn main:app --port 8000`)"
                    resp_placeholder.markdown(answer)
                    thinking_status.update(label="Connection error", state="error", expanded=False)
                    meta = None
                except Exception as e:
                    answer = f"Error: {e}"
                    resp_placeholder.markdown(answer)
                    thinking_status.update(label="Error", state="error", expanded=False)
                    meta = None

                # Eval bar + copyable trace ID
                if meta:
                    render_eval_bar(meta)

            # Save to conversation
            conv["messages"].append({"role": "assistant", "content": answer, "meta": meta})
            st.rerun()


# ── Page: Traces ────────────────────────────────────────────────────────────

elif st.session_state.page == "traces":
    st.title("Trace Explorer")

    search_id = st.text_input("Search by Trace ID", placeholder="Paste a full or partial trace ID...")
    traces = load_traces_from_disk()

    if search_id:
        traces = [t for t in traces if search_id.strip() in t.get("trace_id", "")]

    if not traces:
        st.info("No traces found." if search_id else "No traces yet. Send some messages in Chat first.")
    else:
        trace_options = {
            f'{t.get("trace_id", "?")[:8]}... | {t.get("timestamp", "")[:19]} | {t.get("agent_used", "?")} | {t.get("routing_confidence", "?")}': t
            for t in traces
        }
        selected_label = st.selectbox("Select trace", list(trace_options.keys()))
        selected_trace = trace_options[selected_label]

        # Display trace detail
        st.subheader("Query")
        st.write(selected_trace.get("query", ""))

        st.subheader("Response")
        st.write(selected_trace.get("response", ""))

        col1, col2 = st.columns(2)
        with col1:
            agent = selected_trace.get("agent_used", "?")
            st.markdown(f"**Agent:** {badge(agent.upper(), agent_color(agent))}", unsafe_allow_html=True)
        with col2:
            conf = selected_trace.get("routing_confidence", "?")
            st.markdown(f"**Confidence:** {badge(conf.upper(), conf_color(conf))}", unsafe_allow_html=True)

        st.subheader("Orchestrator Reasoning")
        render_cot(selected_trace.get("cot_reasoning", ""))

        st.subheader("Evaluation Scores")
        scores = selected_trace.get("eval_scores", {})
        col_r, col_f, col_c = st.columns(3)
        for col, key, label in [
            (col_r, "routing_accuracy", "Routing Accuracy"),
            (col_f, "faithfulness", "Faithfulness"),
            (col_c, "resolution_confidence", "Resolution Confidence"),
        ]:
            with col:
                s = scores.get(key, {})
                score_val = s.get("score", "N/A")
                st.metric(label, f"{score_val}/5" if score_val is not None else "N/A")
                st.caption(s.get("reason", ""))

        with st.expander("Raw JSON"):
            st.json(selected_trace)


# ── Page: Batch Eval ────────────────────────────────────────────────────────

elif st.session_state.page == "batch_eval":
    st.title("Batch Evaluation")
    traces = load_traces_from_disk()

    if not traces:
        st.warning("No traces to evaluate.")
    else:
        routing_scores, faith_scores, conf_scores = [], [], []
        agent_counts = {"rag": 0, "web": 0, "both": 0, "escalate": 0}
        conf_counts = {"high": 0, "medium": 0, "low": 0}

        for t in traces:
            agent = t.get("agent_used", "unknown")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            rc = t.get("routing_confidence", "low")
            conf_counts[rc] = conf_counts.get(rc, 0) + 1

            scores = t.get("eval_scores", {})
            for store, key in [(routing_scores, "routing_accuracy"), (faith_scores, "faithfulness"), (conf_scores, "resolution_confidence")]:
                s = scores.get(key, {}).get("score")
                if s is not None and s > 0:
                    store.append(s)

        st.subheader("Mean Scores")
        mean_data = {
            "Routing Accuracy": sum(routing_scores) / len(routing_scores) if routing_scores else 0,
            "Faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0,
            "Resolution Confidence": sum(conf_scores) / len(conf_scores) if conf_scores else 0,
        }
        st.bar_chart(pd.DataFrame({"Score": mean_data}))

        st.subheader("Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Agent distribution**")
            st.bar_chart(pd.DataFrame({"Count": agent_counts}))
        with col2:
            st.write("**Confidence breakdown**")
            st.bar_chart(pd.DataFrame({"Count": conf_counts}))

        st.metric("Total traces", len(traces))


# ── Page: Prompt Manager ───────────────────────────────────────────────────

elif st.session_state.page == "prompts":
    st.title("Prompt Manager")

    PROMPT_KEYS = ["orchestrator", "rag_agent", "web_agent", "judge_routing", "judge_faithfulness", "judge_confidence"]
    selected_key = st.selectbox("Prompt key", PROMPT_KEYS)

    if selected_key:
        try:
            resp = requests.get(f"{API_BASE}/prompts/{selected_key}", timeout=5)
            active = resp.json() if resp.status_code == 200 else None
        except Exception:
            active = None

        if active:
            st.subheader(f"Active: {selected_key} v{active.get('version', '?')}")
            new_text = st.text_area("Prompt text", value=active.get("prompt_text", ""), height=300)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Save new version"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/prompts/{selected_key}",
                            json={"prompt_text": new_text, "created_by": "dashboard", "activate": False},
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.success(f"Saved v{r.json().get('version')}")
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(str(e))

            with col2:
                if st.button("Save and activate"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/prompts/{selected_key}",
                            json={"prompt_text": new_text, "created_by": "dashboard", "activate": True},
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.success(f"Activated v{r.json().get('version')}")
                            st.rerun()
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(str(e))

            with col3:
                if st.button("Rollback"):
                    try:
                        r = requests.post(f"{API_BASE}/prompts/{selected_key}/rollback", timeout=10)
                        if r.status_code == 200:
                            st.toast(f"Rolled back to v{r.json().get('version')}")
                            st.rerun()
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(str(e))

            # Version history table
            st.subheader("Version history")
            try:
                resp = requests.get(f"{API_BASE}/prompts/{selected_key}/history", timeout=5)
                if resp.status_code == 200:
                    versions = resp.json()
                    for v in versions:
                        cols = st.columns([1, 1, 2, 1, 1])
                        cols[0].write(f"v{v['version']}")
                        cols[1].write(v.get("created_by", ""))
                        cols[2].write(str(v.get("created_at", ""))[:19])
                        cols[3].write("ACTIVE" if v.get("is_active") else "")
                        if not v.get("is_active"):
                            if cols[4].button("Activate", key=f"act_{v['version']}"):
                                try:
                                    r = requests.post(f"{API_BASE}/prompts/{selected_key}/activate/{v['version']}", timeout=10)
                                    if r.status_code == 200:
                                        st.rerun()
                                except Exception as e:
                                    st.error(str(e))
            except Exception as e:
                st.warning(f"Could not load history: {e}")
        else:
            st.info("Start the API server to manage prompts.")
