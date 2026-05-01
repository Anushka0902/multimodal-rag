"""
Streamlit UI for the Multi-Modal RAG System.
Run:  streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Multi-Modal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .react-btn {
        background: none;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 20px;
        padding: 4px 12px;
        cursor: pointer;
        font-size: 13px;
        color: inherit;
        transition: all 0.2s;
    }
    .react-btn:hover { background: rgba(128,128,128,0.1); }
    .history-item {
        padding: 6px 10px;
        border-radius: 8px;
        margin-bottom: 4px;
        font-size: 13px;
        cursor: pointer;
        border: 1px solid rgba(128,128,128,0.2);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .relevance-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 6px;
    }
    .rel-high { background: #e8f5e9; color: #2e7d32; }
    .rel-mid  { background: #fff8e1; color: #f57f17; }
    .rel-low  { background: #fce4ec; color: #c62828; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────
@st.cache_resource
def get_rag():
    from src.orchestrator import RAGOrchestrator
    return RAGOrchestrator()

rag = get_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "reactions" not in st.session_state:
    st.session_state.reactions = {}
if "all_chats" not in st.session_state:
    # list of saved chat sessions: [{title, messages, reactions}]
    st.session_state.all_chats = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Multi-Modal RAG")
    st.caption("Chat with PDFs, images & data together")

    st.divider()
    st.subheader("📂 Upload documents")
    uploaded = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "webp", "csv", "tsv"],
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            with st.spinner(f"Ingesting {f.name} …"):
                result = rag.ingest_file(f.read(), f.name)
            if result["success"]:
                st.success(f"✅ {f.name} — {result['vectors_stored']} vectors")
            else:
                st.error(f"❌ {f.name}: {result['error']}")

    st.divider()
    st.subheader("🔍 Search filters")
    doc_type_filter = st.selectbox("Document type", ["All", "PDF", "Image", "CSV"])
    doc_type_filter = None if doc_type_filter == "All" else doc_type_filter.lower()
    year_filter = st.number_input("Year filter (0 = any)", min_value=0, max_value=2030, value=0)
    year_filter = int(year_filter) if year_filter > 0 else None
    alpha = st.slider("Hybrid alpha (0=keyword, 1=semantic)", 0.0, 1.0, 0.5, 0.05)

    st.divider()

    # ── New Chat + Clear History buttons side by side ──────────────────
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✏️ New Chat", use_container_width=True):
            # Save current chat to history before clearing
            if st.session_state.messages:
                first_user = next(
                    (m["content"] for m in st.session_state.messages if m["role"] == "user"),
                    "Chat",
                )
                st.session_state.all_chats.append({
                    "title": first_user[:40] + ("…" if len(first_user) > 40 else ""),
                    "messages": list(st.session_state.messages),
                    "reactions": dict(st.session_state.reactions),
                })
            st.session_state.messages = []
            st.session_state.reactions = {}
            st.session_state.show_history = False
            st.rerun()

    with btn_col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.reactions = {}
            st.rerun()

    # ── Chat history toggle ────────────────────────────────────────────
    if st.session_state.all_chats:
        if st.button(
            "📋 Hide history" if st.session_state.show_history else "📋 Show history",
            use_container_width=True,
        ):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()

        if st.session_state.show_history:
            st.markdown("**Past chats**")
            for idx, chat in enumerate(reversed(st.session_state.all_chats)):
                real_idx = len(st.session_state.all_chats) - 1 - idx
                if st.button(
                    f"💬 {chat['title']}",
                    key=f"history_{real_idx}",
                    use_container_width=True,
                ):
                    # Load this chat back into the active session
                    st.session_state.messages = list(chat["messages"])
                    st.session_state.reactions = dict(chat["reactions"])
                    st.session_state.show_history = False
                    st.rerun()

    st.divider()

    # Relevance score legend instead of vector count
    st.markdown("**Relevance score guide**")
    st.markdown("""
- 🟢 **≥ 0.70** — High relevance
- 🟡 **0.40 – 0.69** — Moderate relevance
- 🔴 **< 0.40** — Low relevance
""")

# ── Main tabs ──────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])

# ── Chat tab ───────────────────────────────────────────────────────────
with tab_chat:

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                # Show relevance score prominently
                avg_score = sum(s.score for s in msg["sources"]) / len(msg["sources"])
                if avg_score >= 0.70:
                    rel_label = "🟢 High relevance"
                    rel_color = "#2e7d32"
                elif avg_score >= 0.40:
                    rel_label = "🟡 Moderate relevance"
                    rel_color = "#f57f17"
                else:
                    rel_label = "🔴 Low relevance"
                    rel_color = "#c62828"

                st.markdown(
                    f"<span style='font-size:12px; color:{rel_color}; font-weight:600;'>"
                    f"{rel_label} ({avg_score:.0%} match)</span>",
                    unsafe_allow_html=True,
                )

                with st.expander("📚 Sources", expanded=False):
                    for s in msg["sources"]:
                        score_color = "#2e7d32" if s.score >= 0.7 else "#f57f17" if s.score >= 0.4 else "#c62828"
                        st.markdown(
                            f"**{s.source}**"
                            + (f" · page {s.page}" if s.page else "")
                            + f" · `{s.doc_type}`"
                            + f" <span style='color:{score_color}; font-size:12px;'>● {s.score:.0%} match</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(s.content[:300] + "…")

            # Reaction bar
            if msg["role"] == "assistant":
                reaction = st.session_state.reactions.get(i)
                col1, col2, col3, col4 = st.columns([1, 1, 1, 8])

                with col1:
                    liked = reaction == "liked"
                    if st.button(
                        "👍 Helpful" if liked else "👍",
                        key=f"like_{i}",
                        help="Mark as helpful",
                        type="primary" if liked else "secondary",
                    ):
                        st.session_state.reactions[i] = None if liked else "liked"
                        st.rerun()

                with col2:
                    disliked = reaction == "disliked"
                    if st.button(
                        "👎 Not helpful" if disliked else "👎",
                        key=f"dislike_{i}",
                        help="Mark as not helpful",
                        type="primary" if disliked else "secondary",
                    ):
                        st.session_state.reactions[i] = None if disliked else "disliked"
                        st.rerun()

                with col3:
                    if st.button("🔗 Share", key=f"share_{i}", help="Copy answer"):
                        st.session_state[f"show_copy_{i}"] = not st.session_state.get(f"show_copy_{i}", False)
                        st.rerun()

                if st.session_state.get(f"show_copy_{i}"):
                    st.code(msg["content"], language=None)
                    st.caption("👆 Click the copy icon in the top-right corner of the box above")
                    if st.button("✕ Close", key=f"close_copy_{i}"):
                        st.session_state[f"show_copy_{i}"] = False
                        st.rerun()

    # Suggested follow-ups
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.markdown("**Suggested follow-ups:**")
        suggestions = [
            "Can you explain more about this?",
            "What are the key takeaways?",
            "Give me a summary in simple terms.",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                    st.session_state["_pending_prompt"] = suggestion
                    st.rerun()

    pending = st.session_state.pop("_pending_prompt", None)
    user_input = pending or st.chat_input("Ask anything about your documents …")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                response, metrics = rag.query(
                    user_input,
                    doc_type_filter=doc_type_filter,
                    year_filter=year_filter,
                    alpha=alpha,
                )
            st.markdown(response.answer)

            if response.sources:
                avg_score = sum(s.score for s in response.sources) / len(response.sources)
                if avg_score >= 0.70:
                    rel_label, rel_color = "🟢 High relevance", "#2e7d32"
                elif avg_score >= 0.40:
                    rel_label, rel_color = "🟡 Moderate relevance", "#f57f17"
                else:
                    rel_label, rel_color = "🔴 Low relevance", "#c62828"

                st.markdown(
                    f"<span style='font-size:12px; color:{rel_color}; font-weight:600;'>"
                    f"{rel_label} ({avg_score:.0%} match)</span>",
                    unsafe_allow_html=True,
                )

                with st.expander("📚 Sources", expanded=False):
                    for s in response.sources:
                        score_color = "#2e7d32" if s.score >= 0.7 else "#f57f17" if s.score >= 0.4 else "#c62828"
                        st.markdown(
                            f"**{s.source}**"
                            + (f" · page {s.page}" if s.page else "")
                            + f" · `{s.doc_type}`"
                            + f" <span style='color:{score_color}; font-size:12px;'>● {s.score:.0%} match</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(s.content[:300] + "…")

        msg_index = len(st.session_state.messages)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })
        st.session_state.reactions[msg_index] = None
        st.rerun()

# ── Eval dashboard tab ─────────────────────────────────────────────────
with tab_eval:
    summary = rag.eval_summary()
    history = rag.eval_history()

    if not history:
        st.info("Ask at least one question to see evaluation metrics.")
    else:
        # ── Helpful vs Not Helpful chart ───────────────────────────────
        st.subheader("👍 Helpful vs 👎 Not Helpful")
        liked_count = sum(1 for v in st.session_state.reactions.values() if v == "liked")
        disliked_count = sum(1 for v in st.session_state.reactions.values() if v == "disliked")
        no_reaction = sum(1 for v in st.session_state.reactions.values() if v is None)

        fig_feedback = go.Figure(data=[
            go.Bar(name="👍 Helpful", x=["Feedback"], y=[liked_count],
                   marker_color="#4caf50"),
            go.Bar(name="👎 Not helpful", x=["Feedback"], y=[disliked_count],
                   marker_color="#e91e63"),
            go.Bar(name="No reaction", x=["Feedback"], y=[no_reaction],
                   marker_color="#90a4ae"),
        ])
        fig_feedback.update_layout(
            barmode="group", height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_feedback, use_container_width=True)

        st.divider()

        # ── Relevance score over queries ───────────────────────────────
        st.subheader("📈 Relevance score per query")
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(
            y=[round(m.mean_relevance_score, 3) for m in history],
            mode="lines+markers",
            name="Relevance score",
            line=dict(color="#1976d2"),
            marker=dict(size=8),
        ))
        fig_rel.add_hline(y=0.70, line_dash="dash", line_color="#4caf50",
                          annotation_text="High ≥0.70")
        fig_rel.add_hline(y=0.40, line_dash="dash", line_color="#f57f17",
                          annotation_text="Moderate ≥0.40")
        fig_rel.update_layout(
            xaxis_title="Query #", yaxis_title="Relevance score",
            yaxis=dict(range=[0, 1]),
            height=300, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_rel, use_container_width=True)

        st.divider()

        # ── Per-query table ────────────────────────────────────────────
        st.subheader("Per-query breakdown")
        rows = [
            {
                "Query": m.query[:60] + ("…" if len(m.query) > 60 else ""),
                "Relevance": round(m.mean_relevance_score, 3),
                "Faithfulness": round(m.faithfulness_score, 3),
                "Chunks used": m.num_chunks_used,
                "Latency (ms)": sum(m.latency_ms.values()),
            }
            for m in history
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)