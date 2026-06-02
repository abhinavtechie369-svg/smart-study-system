import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="Smart Study System", layout="wide")

import time
import pandas as pd
import plotly.express as px # type: ignore

from model import get_model_parameters, predict_score
from utils.helper import analyze_performance


# -------------------------------
# LOAD CSS
# -------------------------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# -------------------------------
# HEADER (PREMIUM LOOK)
# -------------------------------
st.markdown("""
<h1>🚀 Smart Study System</h1>
<p style='text-align:center;'>AI-powered performance prediction & planning</p>
""", unsafe_allow_html=True)


# -------------------------------
# SIDEBAR INPUT
# -------------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    study = st.slider("Study Hours", 0.0, 12.0, 2.0)
    sleep = st.slider("Sleep Hours", 0.0, 10.0, 6.0)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
    previous = st.slider("Previous Score", 0.0, 100.0, 60.0)

    goal = st.number_input("Target Score", 0.0, 100.0, 75.0)


# -------------------------------
# MAIN BUTTON
# -------------------------------
if st.button("🚀 Predict Now", use_container_width=True):

    # -------- Progress Animation --------
    progress = st.progress(0)
    status = st.empty()

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

        if i < 30:
            status.text("Analyzing study pattern...")
        elif i < 70:
            status.text("Training intelligent model...")
        else:
            status.text("Generating insights...")

    progress.empty()
    status.empty()

    # -------- Prediction --------
    score = predict_score(study, sleep, attendance, previous)
    category, advice = analyze_performance(score)

    # -------- MLR Goal Logic --------
    m, b = get_model_parameters()

    m_study = m[0]
    m_sleep = m[1]
    m_att = m[2]
    m_prev = m[3]

    required_hours = (goal - (b + m_sleep*sleep + m_att*attendance + m_prev*previous)) / m_study

    # -------------------------------
    # HERO RESULT (CENTER)
    # -------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align:center; padding:30px;'>
        <h1 style='font-size:60px; color:#22c55e;'>{score:.2f}</h1>
        <p style='font-size:18px;'>Predicted Score</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### Performance: {category}")

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # METRICS ROW
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Score", f"{score:.2f}")
    col2.metric("Performance", category)
    col3.metric("Required Study Hours", f"{required_hours:.2f}")

    if required_hours < 0:
        st.warning("⚠️ Target already achievable!")

    st.success(advice)

    # -------------------------------
    # GRAPH (PREMIUM)
    # -------------------------------
    data = pd.read_csv("data/student_data.csv")
    X = data['study_hours']

    predictions = [
        predict_score(h, sleep, attendance, previous)
        for h in X
    ]

    df = pd.DataFrame({
        "Study Hours": X,
        "Predicted Score": predictions
    })

    fig = px.line(
        df,
        x="Study Hours",
        y="Predicted Score",
        title="📈 Study Performance Intelligence",
        template="plotly_dark"
    )

    fig.update_traces(line=dict(width=4))

    fig.add_scatter(
        x=[study],
        y=[score],
        mode='markers',
        marker=dict(size=12),
        name='Your Prediction'
    )

    fig.update_layout(
        title_font_size=22,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
    )

    st.plotly_chart(fig, use_container_width=True)