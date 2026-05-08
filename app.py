"""
SmartPrep AI — Streamlit Application
Personalized Study Planner & Answer Evaluator
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="SmartPrep AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700;800&family=Pacifico&display=swap');

    html, body, [class*="css"] { font-family: 'Nunito', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #f0fff4 40%, #e8f5e9 70%, #e3f2fd 100%) !important;
    }

    /* ── Main content area text defaults ── */
    .stApp, .stApp p, .stApp span, .stApp div,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span {
        color: #004d40;
    }

    /* ── Selectbox / dropdown text ── */
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox [data-testid="stMarkdownContainer"] p,
    div[data-baseweb="select"] span { color: #e0f7fa !important; }
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li span { color: #004d40 !important; }

    /* ── Multiselect text ── */
    .stMultiSelect div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] div { color: #e0f7fa !important; }
    .stMultiSelect [data-baseweb="tag"] span { color: #ffffff !important; }

    /* ── Slider value label ── */
    .stSlider [data-testid="stThumbValue"],
    .stSlider p { color: #00796b !important; font-weight: 700 !important; }

    /* ── Expander header text ── */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    button[kind="expanderHeader"],
    button[kind="expanderHeader"] p {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        color: #00796b !important;
        background: #e0f7fa !important;
        border-radius: 12px !important;
    }
    /* Expander body text */
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] p { color: #004d40 !important; }

    /* ── Text inputs / text areas ── */
    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input {
        color: #004d40 !important;
        background-color: #ffffff !important;
    }

    /* ── Info / Alert boxes ── */
    .stAlert { border-radius: 14px !important; font-family: 'Nunito', sans-serif !important; font-weight: 600 !important; }
    .stAlert p, .stAlert div, .stAlert span { color: inherit !important; }
    div[data-testid="stAlert"] p { color: #004d40 !important; }

    /* ── st.info specifically (light blue bg) ── */
    div[data-testid="stAlert"][data-baseweb="notification"] p { color: #004d40 !important; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #e0f7fa 100%) !important;
        border-right: 3px solid #b2ebf2 !important;
    }
    /* ── Sidebar ALL text — force visible dark teal ── */
    section[data-testid="stSidebar"] * {
        color: #00695c !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Fredoka One', cursive !important;
        color: #00695c !important;
        font-size: 1.6rem !important;
        letter-spacing: 1px;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #00695c !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        color: #00695c !important;
        font-size: 1rem !important;
    }
    /* Radio button selected state */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"] p,
    section[data-testid="stSidebar"] .stRadio label p {
        color: #00695c !important;
        font-weight: 800 !important;
    }
    /* Caption / info box text */
    section[data-testid="stSidebar"] .stAlert p,
    section[data-testid="stSidebar"] .stAlert div {
        color: #00695c !important;
    }

    .main-title {
        font-family: 'Pacifico', cursive !important;
        font-size: 3rem !important;
        background: linear-gradient(90deg, #00bcd4, #26a69a, #66bb6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 0.5rem 0;
        filter: drop-shadow(0px 2px 4px rgba(0,188,212,0.3));
    }
    .sub-title {
        font-family: 'Nunito', sans-serif !important;
        font-size: 1.15rem !important;
        color: #4db6ac !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }

    .metric-box {
        background: linear-gradient(135deg, #26c6da 0%, #00897b 100%);
        color: white !important;
        border-radius: 20px;
        padding: 1.4rem 1rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(38,198,218,0.25);
        font-family: 'Nunito', sans-serif !important;
    }
    .metric-box h2 {
        font-family: 'Fredoka One', cursive !important;
        font-size: 2.2rem !important;
        margin: 0 0 0.2rem 0;
        color: white !important;
    }
    .metric-box p { margin: 0; font-size: 0.9rem; opacity: 0.92; font-weight: 600; color: white !important; }

    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        border-left: 6px solid #26c6da;
        margin-bottom: 1rem;
        box-shadow: 0 4px 18px rgba(38,198,218,0.12);
        font-family: 'Nunito', sans-serif !important;
    }
    .card strong {
        font-family: 'Fredoka One', cursive !important;
        font-size: 1.1rem;
        color: #00897b !important;
        letter-spacing: 0.5px;
    }
    .card small { color: #546e7a !important; font-size: 0.88rem; line-height: 1.5; }

    .weak  { color: #ef5350 !important; font-weight: 800 !important; }
    .strong { color: #26a69a !important; font-weight: 800 !important; }

    h1, h2, h3 {
        font-family: 'Fredoka One', cursive !important;
        color: #00796b !important;
        letter-spacing: 0.5px;
    }

    /* ── Markdown text in main area ── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #004d40 !important;
    }
    [data-testid="stMarkdownContainer"] strong { color: #00695c !important; }

    div.stButton > button {
        background: linear-gradient(135deg, #26c6da 0%, #00897b 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        font-family: 'Fredoka One', cursive !important;
        font-size: 1.05rem !important;
        padding: 0.55rem 2rem !important;
        box-shadow: 0 4px 16px rgba(38,198,218,0.35) !important;
        letter-spacing: 0.5px;
    }

    label, .stSelectbox label, .stSlider label, .stTextArea label,
    .stNumberInput label, .stMultiSelect label {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        color: #00796b !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Fredoka One', cursive !important;
        color: #00897b !important;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #26c6da !important;
        color: #006064 !important;
    }

    .streamlit-expanderHeader {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        color: #00796b !important;
        background: #e0f7fa !important;
        border-radius: 12px !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 0.8rem 1rem;
        box-shadow: 0 3px 12px rgba(38,198,218,0.12);
    }
    [data-testid="stMetricLabel"] { font-family: 'Nunito', sans-serif !important; color: #4db6ac !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { font-family: 'Fredoka One', cursive !important; color: #00796b !important; }
    [data-testid="stMetricDelta"] { color: #26a69a !important; }

    .stDataFrame { border-radius: 16px !important; overflow: hidden; box-shadow: 0 4px 16px rgba(38,198,218,0.1); }

    .stCaption, .stMarkdown small, footer { font-family: 'Nunito', sans-serif !important; color: #546e7a !important; font-weight: 600; }
    hr { border-color: #b2dfdb !important; border-width: 2px !important; border-radius: 2px; }

    /* ── Number input spinners ── */
    .stNumberInput div[data-baseweb="input"] input { color: #004d40 !important; background: #ffffff !important; }

    /* ── Subheader text ── */
    .stApp h2 { color: #00796b !important; }
    .stApp h3 { color: #00796b !important; }

    /* ── Study planner markdown bullets & Recommendations ── */
    .stMarkdown ul li, .stMarkdown ol li { color: #004d40 !important; }

    /* ── All markdown list items (covers Recommendations & Detailed Feedback) ── */
    [data-testid="stMarkdownContainer"] ul li,
    [data-testid="stMarkdownContainer"] ol li,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] li p,
    [data-testid="stMarkdownContainer"] li span,
    [data-testid="stMarkdownContainer"] li strong,
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol {
        color: #004d40 !important;
        font-weight: 600 !important;
    }
    /* Each st.markdown("- item") renders its own container — force all p/text too */
    [data-testid="stMarkdownContainer"] > p,
    [data-testid="stMarkdownContainer"] > div > p {
        color: #004d40 !important;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #e0f7fa; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: #80cbc4; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Playful chart palette ──────────────────────────────────
PALETTE  = ["#26c6da", "#ef5350", "#66bb6a", "#ffa726", "#ab47bc"]
CHART_BG = "#f7fffe"

def style_ax(ax, title=""):
    ax.set_facecolor(CHART_BG)
    ax.figure.set_facecolor(CHART_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b2dfdb")
    ax.spines["bottom"].set_color("#b2dfdb")
    ax.tick_params(colors="#00796b", labelsize=9)
    ax.yaxis.label.set_color("#00796b")
    ax.xaxis.label.set_color("#00796b")
    if title:
        ax.set_title(title, color="#00695c", fontsize=11, fontweight="bold", pad=10)

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
st.sidebar.title("SmartPrep AI 🎓")
st.sidebar.caption("Personalized Study Planner & Answer Evaluator")

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📅 Study Planner",
    "🔍 Difficulty Predictor",
    "📊 Weak Topic Analyzer",
    "✏️ Answer Evaluator",
    "📈 Model Comparison"
])
st.sidebar.markdown("---")
st.sidebar.info("B.Tech 2nd Year | AIML Lab | Jaypee Institute of Information Technology")

# ── Lazy load models ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    if not os.path.exists("models/difficulty_model.pkl"):
        with st.spinner("First run: Training models... (30–60 sec)"):
            import subprocess
            subprocess.run([sys.executable, "train_all.py"], capture_output=True)
    from ml_models.difficulty_predictor import load_difficulty_model
    from ml_models.weak_topic_analyzer import load_kmeans_models, train_kmeans
    from dl_models.answer_evaluator import load_answer_eval_model
    diff_model, diff_vec, diff_le = load_difficulty_model()
    km_model, km_scaler, km_pca = load_kmeans_models()
    ae_model = load_answer_eval_model()
    dfs = {}
    for name, path in [("difficulty", "dataset/difficulty_dataset.csv"),
                        ("students", "dataset/student_performance.csv"),
                        ("answers", "dataset/answer_evaluation.csv")]:
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
    km_result = None
    if km_model and "students" in dfs:
        X = dfs["students"][["Mathematics","Physics","Computer Science","Chemistry","Biology"]].values
        from sklearn.preprocessing import StandardScaler
        labels = km_model.labels_
        cluster_means = []
        for c in range(3):
            mask = labels == c
            mean_score = dfs["students"][["Mathematics","Physics","Computer Science","Chemistry","Biology"]][mask].mean().mean()
            cluster_means.append((c, mean_score))
        sorted_clusters = sorted(cluster_means, key=lambda x: x[1])
        km_result = {
            "cluster_map": {sorted_clusters[0][0]: "Weak", sorted_clusters[1][0]: "Average", sorted_clusters[2][0]: "Strong"},
            "labels": labels
        }
    return diff_model, diff_vec, diff_le, km_model, km_scaler, km_pca, km_result, ae_model, dfs


# ══════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-title">🎓 SmartPrep AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">✨ Personalized Study Planner & Answer Evaluator powered by AI/ML ✨</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box"><h2>5</h2><p>AI Algorithms</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><h2>3</h2><p>ML Models</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><h2>300+</h2><p>Training Samples</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box"><h2>5</h2><p>Subjects</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔧 System Modules")
    cols = st.columns(2)
    modules = [
        ("📅 Study Planner", "Generates optimized study schedules using BFS, DFS, UCS, A*, Hill Climbing search algorithms based on your subjects and weak topics."),
        ("🔍 Difficulty Predictor", "Predicts question difficulty (Easy/Medium/Hard) using Logistic Regression, Random Forest, and SVM with TF-IDF features."),
        ("📊 Weak Topic Analyzer", "Identifies your weak, moderate, and strong subjects using K-Means clustering on performance data."),
        ("✏️ Answer Evaluator", "Evaluates subjective answers using NLP (TF-IDF + Cosine Similarity) and provides detailed feedback with score prediction.")
    ]
    for i, (title, desc) in enumerate(modules):
        with cols[i % 2]:
            st.markdown(f'<div class="card"><strong>{title}</strong><br><small>{desc}</small></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Project by: Namya Jain | Ujwal Parashar | Ananya Sachdeva | Shaurya Gupta | Submitted to: Dr. Akansha Singh")

# ══════════════════════════════════════════════════════════
# PAGE: STUDY PLANNER
# ══════════════════════════════════════════════════════════
elif page == "📅 Study Planner":
    st.title("📅 Personalized Study Planner")
    from dataset.generate_datasets import TOPIC_TREE
    from planner.search_algorithms import generate_study_plan, compare_all_algorithms
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Configure Your Plan")
        subject = st.selectbox("Select Subject", list(TOPIC_TREE.keys()))
        topics = TOPIC_TREE[subject]
        weak_topics = st.multiselect("Mark Weak Topics", topics)
        exam_days = st.slider("Days Until Exam", 3, 21, 7)
        algorithm = st.selectbox("Search Algorithm", ["A*", "UCS", "BFS", "DFS", "Hill Climbing"])
        st.markdown("**Algorithm Info:**")
        algo_desc = {
            "A*": "Combines actual cost + heuristic. Optimal & efficient.",
            "UCS": "Expands minimum cost node. Optimal for weighted graphs.",
            "BFS": "Level-by-level. Shortest path in unweighted graphs.",
            "DFS": "Depth-first. Memory efficient, may miss optimal path.",
            "Hill Climbing": "Greedy local search. Fast but may get stuck."
        }
        st.info(algo_desc[algorithm])
        generate = st.button("🚀 Generate Study Plan", type="primary")
    with col2:
        if generate:
            plan = generate_study_plan(subject, topics, weak_topics, exam_days, algorithm)
            st.subheader(f"📋 Study Plan: {subject} ({algorithm})")
            st.markdown(f"**Total Topics:** {plan['total_topics']} | **Exam Days:** {exam_days}")
            for day_info in plan["daily_plan"]:
                with st.expander(f"Day {day_info['day']} — {day_info['focus']} ({day_info['hours']}h)", expanded=day_info["day"] <= 3):
                    for t in day_info["topics"]:
                        if t in weak_topics:
                            st.markdown(f"🔴 **{t}** *(Weak — Focus More)*")
                        else:
                            st.markdown(f"🟢 {t}")
            st.markdown("---")
            st.subheader("⚡ Algorithm Comparison")
            comparison = compare_all_algorithms(topics, weak_topics)
            comp_df = pd.DataFrame([{
                "Algorithm": r["algorithm"],
                "Path Length": len(r["path"]),
                "Nodes Explored": r["nodes_explored"],
                "Time (ms)": r["time_ms"]
            } for r in comparison])
            st.dataframe(comp_df, use_container_width=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=CHART_BG)
            axes[0].bar(comp_df["Algorithm"], comp_df["Nodes Explored"], color=PALETTE, edgecolor="white", linewidth=1.2)
            style_ax(axes[0], "Nodes Explored per Algorithm")
            axes[0].set_ylabel("Nodes")
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha='right')
            axes[1].bar(comp_df["Algorithm"], comp_df["Time (ms)"], color=PALETTE, edgecolor="white", linewidth=1.2)
            style_ax(axes[1], "Execution Time (ms)")
            axes[1].set_ylabel("Milliseconds")
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE: DIFFICULTY PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🔍 Difficulty Predictor":
    st.title("🔍 Question Difficulty Predictor")
    diff_model, diff_vec, diff_le, *_ = load_all_models()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Enter Question")
        question = st.text_area("Question Text", "Explain the process of DNA replication with all key enzymes.", height=120)
        subject = st.selectbox("Subject", ["Mathematics", "Physics", "Computer Science", "Chemistry", "Biology"])
        predict_btn = st.button("🔮 Predict Difficulty", type="primary")
    with col2:
        if predict_btn and diff_model:
            from ml_models.difficulty_predictor import predict_difficulty
            result = predict_difficulty(question, subject, diff_model, diff_vec, diff_le)
            color_map = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
            emoji = color_map.get(result["predicted_difficulty"], "⚪")
            st.subheader("Prediction Result")
            st.markdown(f"## {emoji} {result['predicted_difficulty']}")
            st.metric("Confidence", f"{result['confidence']}%")
            if result["probabilities"]:
                st.subheader("Probability Distribution")
                probs = result["probabilities"]
                fig, ax = plt.subplots(figsize=(6, 3), facecolor=CHART_BG)
                bars = ax.barh(list(probs.keys()), list(probs.values()),
                               color=["#66bb6a", "#ffa726", "#ef5350"], edgecolor="white", linewidth=1.2)
                style_ax(ax, "Difficulty Probabilities")
                ax.set_xlabel("Probability (%)")
                for bar, val in zip(bars, probs.values()):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f"{val:.1f}%", va="center", color="#00796b", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
        elif predict_btn:
            st.warning("Models not loaded. Please run train_all.py first.")
    st.markdown("---")
    st.subheader("📋 Try Sample Questions")
    samples = [
        ("What is 2+2?", "Mathematics"),
        ("Solve the quadratic equation x^2-5x+6=0.", "Mathematics"),
        ("Derive Maxwell's equations from first principles.", "Physics"),
        ("What is an algorithm?", "Computer Science"),
        ("Explain quantum tunneling.", "Physics")
    ]
    if diff_model:
        from ml_models.difficulty_predictor import predict_difficulty
        sample_results = []
        for q, s in samples:
            r = predict_difficulty(q, s, diff_model, diff_vec, diff_le)
            sample_results.append({"Question": q[:60]+"...", "Subject": s,
                                   "Difficulty": r["predicted_difficulty"],
                                   "Confidence": f"{r['confidence']}%"})
        st.dataframe(pd.DataFrame(sample_results), use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: WEAK TOPIC ANALYZER
# ══════════════════════════════════════════════════════════
elif page == "📊 Weak Topic Analyzer":
    st.title("📊 Weak Topic Analyzer")
    _, _, _, km_model, km_scaler, km_pca, km_result, _, dfs = load_all_models()
    SUBJECTS = ["Mathematics", "Physics", "Computer Science", "Chemistry", "Biology"]
    tab1, tab2 = st.tabs(["🧑‍🎓 My Analysis", "🏫 Class Overview"])
    with tab1:
        st.subheader("Enter Your Scores")
        cols = st.columns(5)
        scores = {}
        defaults = [55, 72, 40, 65, 48]
        for i, (col, subj) in enumerate(zip(cols, SUBJECTS)):
            with col:
                scores[subj] = st.number_input(subj, 0, 100, defaults[i])
        if st.button("🔍 Analyze My Topics", type="primary"):
            if km_model and km_result:
                from ml_models.weak_topic_analyzer import identify_weak_topics_for_student
                result = identify_weak_topics_for_student(scores, km_model, km_scaler, km_result["cluster_map"])
                st.markdown(f"### Performance Group: **{result['performance_group']}**")
                cols = st.columns(3)
                with cols[0]:
                    st.error(f"🔴 Weak: {', '.join(result['weak_subjects']) or 'None'}")
                with cols[1]:
                    st.warning(f"🟡 Moderate: {', '.join(result['moderate_subjects']) or 'None'}")
                with cols[2]:
                    st.success(f"🟢 Strong: {', '.join(result['strong_subjects']) or 'None'}")
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=CHART_BG)
                bar_colors = ["#ef5350" if scores[s] < 50 else "#ffa726" if scores[s] < 65 else "#66bb6a" for s in SUBJECTS]
                bars = ax.bar(SUBJECTS, [scores[s] for s in SUBJECTS], color=bar_colors, edgecolor="white", linewidth=1.5)
                ax.axhline(y=50, color="#ef5350", linestyle="--", alpha=0.7, label="Weak threshold (50)")
                ax.axhline(y=65, color="#ffa726", linestyle="--", alpha=0.7, label="Moderate threshold (65)")
                style_ax(ax, "Subject-wise Performance Analysis")
                ax.set_ylabel("Score (%)")
                ax.legend(fontsize=9)
                ax.set_ylim(0, 110)
                for bar, s in zip(bars, SUBJECTS):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                            f"{scores[s]}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#00695c")
                plt.tight_layout()
                st.pyplot(fig)
                if result["recommendations"]:
                    st.subheader("📌 Recommendations")
                    for rec in result["recommendations"]:
                        st.markdown(f"- {rec}")
    with tab2:
        st.subheader("Class-wide Performance Clusters")
        if km_model and "students" in dfs:
            df_students = dfs["students"]
            labels = km_result["labels"]
            cluster_map = km_result["cluster_map"]
            from ml_models.weak_topic_analyzer import get_class_weak_topic_analysis
            class_analysis = get_class_weak_topic_analysis(df_students, labels, cluster_map)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=CHART_BG)
            mean_scores = [class_analysis[s]["mean_score"] for s in SUBJECTS]
            pct_weak = [class_analysis[s]["pct_weak"] for s in SUBJECTS]
            axes[0].bar(SUBJECTS, mean_scores, color="#26c6da", alpha=0.9, edgecolor="white", linewidth=1.2)
            style_ax(axes[0], "Class Average Score by Subject")
            axes[0].set_ylabel("Average Score")
            axes[0].set_ylim(0, 100)
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha='right')
            axes[1].bar(SUBJECTS, pct_weak, color="#ef5350", alpha=0.9, edgecolor="white", linewidth=1.2)
            style_ax(axes[1], "% Students with Weak Performance (<50)")
            axes[1].set_ylabel("Percentage (%)")
            axes[1].set_ylim(0, 100)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            cluster_counts = pd.Series(labels).map(cluster_map).value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor=CHART_BG)
            ax2.set_facecolor(CHART_BG)
            ax2.pie(cluster_counts.values, labels=cluster_counts.index,
                    autopct='%1.1f%%', colors=["#ef5350","#ffa726","#66bb6a"],
                    startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
            ax2.set_title("Student Performance Cluster Distribution", color="#00695c", fontweight="bold")
            st.pyplot(fig2)

# ══════════════════════════════════════════════════════════
# PAGE: ANSWER EVALUATOR
# ══════════════════════════════════════════════════════════
elif page == "✏️ Answer Evaluator":
    st.title("✏️ Subjective Answer Evaluator")
    _, _, _, _, _, _, _, ae_model, _ = load_all_models()
    from dl_models.answer_evaluator import evaluate_answer
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Reference Answer")
        reference = st.text_area("Reference / Model Answer",
            "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce oxygen and glucose. The equation is 6CO2 + 6H2O + light → C6H12O6 + 6O2.",
            height=150)
        st.subheader("Student Answer")
        student = st.text_area("Student's Answer",
            "Plants make food from sunlight, water and CO2 and release oxygen.",
            height=120)
        max_score = st.slider("Maximum Score", 1, 10, 5)
        evaluate_btn = st.button("📝 Evaluate Answer", type="primary")
    with col2:
        if evaluate_btn:
            result = evaluate_answer(student, reference, max_score, ae_model)
            score_pct = result["percentage"]
            if score_pct >= 80:
                st.success(f"### {result['grade']}")
            elif score_pct >= 60:
                st.info(f"### {result['grade']}")
            elif score_pct >= 40:
                st.warning(f"### {result['grade']}")
            else:
                st.error(f"### {result['grade']}")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Score", f"{result['predicted_score']}/{max_score}")
            col_b.metric("Percentage", f"{result['percentage']}%")
            col_c.metric("Similarity", f"{result['cosine_similarity']}%")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor=CHART_BG)
            ax.set_facecolor(CHART_BG)
            theta = np.linspace(0, np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), "-", color="#e0f7fa", lw=12, solid_capstyle="round")
            fill_angle = np.pi * (score_pct / 100)
            theta_fill = np.linspace(0, fill_angle, 100)
            gauge_color = "#66bb6a" if score_pct >= 60 else "#ffa726" if score_pct >= 40 else "#ef5350"
            ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=gauge_color, lw=12, solid_capstyle="round")
            ax.text(0, -0.2, f"{score_pct:.1f}%", ha="center", va="center",
                    fontsize=24, fontweight="bold", color=gauge_color)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.5, 1.2)
            ax.axis("off")
            ax.set_title("Evaluation Score", fontsize=12, color="#00796b", fontweight="bold")
            st.pyplot(fig)
            st.subheader("📝 Detailed Feedback")
            for fb in result["feedback"]:
                st.markdown(f"- {fb}")
    st.markdown("---")
    st.subheader("📊 Dataset Evaluation Results")
    if ae_model and "answers" in load_all_models()[-1]:
        from preprocessing.text_processor import prepare_answer_eval_features
        df_ans = load_all_models()[-1]["answers"]
        df_eval = prepare_answer_eval_features(df_ans)
        sample = df_eval.sample(10, random_state=42)[["prompt","student_answer","score","cosine_similarity","predicted_score"]].copy()
        sample.columns = ["Prompt","Student Answer","Actual Score","Similarity (%)","Predicted Score"]
        sample["Similarity (%)"] = sample["Similarity (%)"].apply(lambda x: f"{x*100:.1f}")
        sample["Student Answer"] = sample["Student Answer"].apply(lambda x: x[:60]+"...")
        st.dataframe(sample, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("📈 Model Performance Comparison")
    with st.spinner("Loading and evaluating models..."):
        diff_model, diff_vec, diff_le, *_ = load_all_models()
        from dataset.generate_datasets import generate_difficulty_dataset
        from ml_models.difficulty_predictor import train_difficulty_models
        import os
        if os.path.exists("dataset/difficulty_dataset.csv"):
            df_diff = pd.read_csv("dataset/difficulty_dataset.csv")
            results, _, _, _, best = train_difficulty_models(df_diff)
            st.subheader("Difficulty Prediction — Model Comparison")
            metrics_df = pd.DataFrame([{
                "Model": name,
                "Accuracy (%)": m["accuracy"],
                "Precision (%)": m["precision"],
                "Recall (%)": m["recall"],
                "F1-Score (%)": m["f1_score"],
                "CV Mean (%)": m["cv_mean"],
                "CV Std (±%)": m["cv_std"]
            } for name, m in results.items()])
            st.dataframe(metrics_df.style.highlight_max(
                subset=["Accuracy (%)", "F1-Score (%)"], color="#b2dfdb"),
                use_container_width=True)
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=CHART_BG)
            model_names = list(results.keys())
            bar_colors = ["#26c6da", "#66bb6a", "#ab47bc"]
            for ax, metric, title in [
                (axes[0,0], "accuracy", "Accuracy (%)"),
                (axes[0,1], "f1_score", "F1-Score (%)"),
                (axes[1,0], "precision", "Precision (%)"),
                (axes[1,1], "recall", "Recall (%)")
            ]:
                vals = [results[m][metric] for m in model_names]
                bars = ax.bar(model_names, vals, color=bar_colors, alpha=0.9, edgecolor="white", linewidth=1.5)
                style_ax(ax, title)
                ax.set_ylim(0, 115)
                ax.set_ylabel("%")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f"{v}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#00695c")
            plt.suptitle("Difficulty Prediction Model Comparison", fontsize=14, fontweight="bold", color="#00796b")
            plt.tight_layout()
            st.pyplot(fig)
            st.subheader("Confusion Matrices")
            cols = st.columns(3)
            for col, name in zip(cols, model_names):
                with col:
                    cm = np.array(results[name]["confusion_matrix"])
                    fig2, ax2 = plt.subplots(figsize=(4, 3), facecolor=CHART_BG)
                    ax2.set_facecolor(CHART_BG)
                    im = ax2.imshow(cm, interpolation="nearest", cmap="YlGn")
                    style_ax(ax2, name)
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    ax2.set_xticks([0,1,2])
                    ax2.set_yticks([0,1,2])
                    ax2.set_xticklabels(["Easy","Med","Hard"], fontsize=8)
                    ax2.set_yticklabels(["Easy","Med","Hard"], fontsize=8)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax2.text(j, i, cm[i,j], ha="center", va="center",
                                     color="white" if cm[i,j] > cm.max()/2 else "#00695c", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig2)
