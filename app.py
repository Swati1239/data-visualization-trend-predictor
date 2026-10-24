"""
╔══════════════════════════════════════════════════════════════════╗
║         DATA VISUALIZATION TREND PREDICTOR                     ║
║  A Streamlit app for curve fitting and trend prediction         ║
║  Supports Linear & Polynomial regression with interactive UI   ║
╚══════════════════════════════════════════════════════════════════╝

Run with:  streamlit run app.py

Dependencies: streamlit, numpy, plotly, pandas
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import io
import base64

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & THEME
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Viz Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM CSS – Modern dark-mode-friendly styling
# ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        /* FIX SELECTBOX TEXT */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    color: black !important;
}

/* Selected value text */
section[data-testid="stSidebar"] .stSelectbox span {
    color: black !important;
}

/* Dropdown menu items */
div[role="listbox"] div {
    color: black !important;
}
        /* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #0f172a);
}

/* FORCE ALL SIDEBAR TEXT WHITE */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Labels (Manual Entry, CSV Upload, etc.) */
section[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
    font-weight: 500;
}

/* Radio + select text */
section[data-testid="stSidebar"] .stRadio div,
section[data-testid="stSidebar"] .stSelectbox div {
    color: white !important;
}

/* Input boxes */
section[data-testid="stSidebar"] input {
    color: black !important;
    background-color: white !important;
}
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    /* ── Section Cards ── */
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .section-card h3 {
        margin-top: 0;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* ── Metric Cards ── */
    .metric-card {
       background: #f9fafb !important;   /* light background */
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    }
    .metric-card .metric-label {
        color: #374151 !important;   /* dark gray */
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-card .metric-value {
        color: #a78bfa;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* ── Equation Display ── */
    .equation-box {
         background: #ecfeff !important;   /* light cyan */
    border: 1px solid #06b6d4;
    border-radius: 12px;
    padding: 1.2rem;

    font-family: 'Courier New', monospace;
    font-size: 1.1rem;

    color: #0f172a !important;   /* DARK TEXT (important) */
    font-weight: 600;

    text-align: center;
    }

    /* ── Prediction Result ── */
    .prediction-result {
        background: #fef3c7 !important;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-result .pred-label {
         color: #92400e !important;   /* dark brown */
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .prediction-result .pred-value {
        color: #b45309 !important;   /* strong visible */
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* ── Error Card ── */
    .error-card {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #fca5a5;
        margin: 0.5rem 0;
    }

    /* ── Success Card ── */
    .success-card {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #6ee7b7;
        margin: 0.5rem 0;
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30,27,75,0.95) 0%, rgba(15,23,42,0.98) 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Streamlit Element Overrides ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.35) !important;
    }
    /* 🔥 FORCE FIX SELECTBOX (FINAL) */

/* Main selectbox container */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
}

/* Selected value text */
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: black !important;
}

/* Dropdown list items */
div[role="listbox"] * {
    color: black !important;
}

/* Fix input cursor visibility */
section[data-testid="stSidebar"] input {
    color: black !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════


def parse_comma_separated(text: str) -> np.ndarray:
    """
    Parse a comma-separated string of numbers into a numpy array.
    Raises ValueError for non-numeric or empty input.
    """
    if not text or not text.strip():
        raise ValueError("Input is empty. Please enter comma-separated numbers.")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("No valid numbers found. Use commas to separate values.")
    try:
        return np.array([float(p) for p in parts])
    except ValueError:
        raise ValueError(
            f"Non-numeric value detected. Ensure all values are numbers."
        )


def fit_model(x: np.ndarray, y: np.ndarray, model: str, degree: int = 2):
    """
    Fit a regression model to the data.
    Returns: (coefficients, fitted_y_values, x_smooth, y_smooth)
    """
    if model == "Linear Regression":
        degree = 1

    # Fit polynomial (linear is degree 1)
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    # Generate smooth curve for plotting
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min if x_max != x_min else 1.0
    x_smooth = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 300)
    y_smooth = poly(x_smooth)

    return coeffs, poly, x_smooth, y_smooth


def format_equation(coeffs: np.ndarray, degree: int) -> str:
    """
    Format polynomial coefficients into a readable equation string.
    Example: y = 2.50x² + 1.30x + 0.50
    """
    superscripts = {
        0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵",
    }
    terms = []
    n = len(coeffs)

    for i, c in enumerate(coeffs):
        power = n - 1 - i
        coeff_str = f"{abs(c):.4f}"

        if power == 0:
            term = coeff_str
        elif power == 1:
            term = f"{coeff_str}·x"
        else:
            sup = superscripts.get(power, f"^{power}")
            term = f"{coeff_str}·x{sup}"

        if i == 0:
            sign = "- " if c < 0 else ""
        else:
            sign = " - " if c < 0 else " + "

        terms.append(f"{sign}{term}")

    return "y = " + "".join(terms)


def predict_value(poly: np.poly1d, x_val: float) -> float:
    """Predict Y for a given X using the fitted polynomial."""
    return float(poly(x_val))


def create_plot(
    x_data, y_data, x_smooth, y_smooth, equation_str, model_name, pred_x=None, pred_y=None
):
    """
    Create an interactive Plotly figure with data points,
    fitted curve, and optional prediction marker.
    """
    fig = go.Figure()

    # ── Scatter: Original Data Points ──
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            name="Data Points",
            marker=dict(
                size=12,
                color="#a78bfa",
                line=dict(width=2, color="#7c3aed"),
                symbol="circle",
                opacity=0.9,
            ),
            hovertemplate="<b>Data Point</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        )
    )

    # ── Line: Fitted Curve ──
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=f"Fitted Curve ({model_name})",
            line=dict(color="#34d399", width=3, shape="spline"),
            hovertemplate="<b>Fitted</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        )
    )

    # ── Marker: Prediction Point ──
    if pred_x is not None and pred_y is not None:
        fig.add_trace(
            go.Scatter(
                x=[pred_x],
                y=[pred_y],
                mode="markers+text",
                name=f"Prediction (X={pred_x:.2f})",
                marker=dict(
                    size=16,
                    color="#fbbf24",
                    line=dict(width=3, color="#f59e0b"),
                    symbol="star",
                ),
                text=[f"({pred_x:.2f}, {pred_y:.2f})"],
                textposition="top center",
                textfont=dict(color="#fbbf24", size=13, family="Inter"),
                hovertemplate="<b>Prediction</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
            )
        )

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text=f"📈 {model_name} — Trend Analysis",
            font=dict(size=20, color="#e2e8f0", family="Inter"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="X Values", font=dict(size=14, color="#94a3b8")),
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=dict(text="Y Values", font=dict(size=14, color="#94a3b8")),
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            tickfont=dict(color="#94a3b8"),
        ),
        plot_bgcolor="rgba(15,23,42,0.95)",
        paper_bgcolor="rgba(15,23,42,0.95)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color="#cbd5e1", size=12),
            x=0.01,
            y=0.99,
        ),
        margin=dict(l=60, r=30, t=70, b=60),
        hovermode="closest",
        height=520,
    )

    # Add subtle annotation for the equation
    fig.add_annotation(
        text=equation_str,
        xref="paper", yref="paper",
        x=0.5, y=-0.14,
        showarrow=False,
        font=dict(size=12, color="#6ee7b7", family="Courier New"),
        align="center",
    )

    return fig


# ══════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════


def main():
    """Entry point – builds the entire Streamlit UI."""

    # ── Hero Header ──
    st.markdown(
        """
        <div class="hero-header">
            <h1>📈 Data Visualization Trend Predictor</h1>
            <p>Fit curves · Visualize trends · Predict outcomes — all in your browser</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ──────────────────────────────────────────────
    # SIDEBAR – INPUT PANEL
    # ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # ── Data Input Method ──
        st.markdown("### 📥 Data Input")
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "CSV Upload"],
            horizontal=True,
        )

        x_data = None
        y_data = None

        if input_method == "Manual Entry":
            st.markdown(
                '<p style="color:rgba(255,255,255,0.5); font-size:0.85rem;">'
                "Enter comma-separated numeric values</p>",
                unsafe_allow_html=True,
            )
            x_input = st.text_input(
                "X Values",
                placeholder="e.g. 1, 2, 3, 4, 5",
                help="Enter X-axis data points separated by commas.",
            )
            y_input = st.text_input(
                "Y Values",
                placeholder="e.g. 2.1, 4.0, 5.8, 8.2, 10.1",
                help="Enter Y-axis data points separated by commas.",
            )

            # Parse inputs (real-time as user types)
            if x_input and y_input:
                try:
                    x_data = parse_comma_separated(x_input)
                    y_data = parse_comma_separated(y_input)
                except ValueError as e:
                    st.markdown(
                        f'<div class="error-card">⚠️ {e}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="CSV must have at least two numeric columns. "
                     "The first column is X, the second is Y.",
            )
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if df.shape[1] < 2:
                        st.markdown(
                            '<div class="error-card">'
                            "⚠️ CSV must have at least 2 columns (X, Y).</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        # Take first two columns and drop NaNs
                        df = df.iloc[:, :2].dropna()
                        df.columns = ["X", "Y"]
                        x_data = df["X"].values.astype(float)
                        y_data = df["Y"].values.astype(float)
                        st.markdown(
                            f'<div class="success-card">'
                            f"✅ Loaded {len(x_data)} data points from CSV.</div>",
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.markdown(
                        f'<div class="error-card">⚠️ Error reading CSV: {e}</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

        # ── Model Selection ──
        st.markdown("### 🧮 Model Selection")
        model_type = st.selectbox(
            "Regression Type",
            ["Linear Regression", "Polynomial Regression"],
            help="Choose the type of curve to fit to your data.",
        )

        poly_degree = 2
        if model_type == "Polynomial Regression":
            poly_degree = st.slider(
                "Polynomial Degree",
                min_value=2,
                max_value=5,
                value=2,
                help="Higher degrees fit more complex curves but may overfit.",
            )

        st.markdown("---")

        # ── Prediction Input ──
        st.markdown("### 🔮 Prediction")
        predict_x = st.number_input(
            "Enter X value to predict Y",
            value=0.0,
            format="%.4f",
            help="Enter any X value to get the predicted Y from the fitted model.",
        )

        st.markdown("---")
        st.markdown(
            '<div class="footer">Built with Streamlit · NumPy · Plotly</div>',
            unsafe_allow_html=True,
        )

    # ──────────────────────────────────────────────
    # MAIN CONTENT AREA
    # ──────────────────────────────────────────────

    # Validate data before proceeding
    if x_data is None or y_data is None:
        # Show a friendly placeholder state
        st.markdown(
            """
            <div class="section-card" style="text-align:center; padding:4rem 2rem;">
                <h3 style="font-size:1.5rem;">👋 Welcome!</h3>
                <p style="color:rgba(255,255,255,0.55); font-size:1rem; max-width:500px; margin:1rem auto;">
                    Enter your data points in the sidebar to get started.
                    You can type comma-separated values or upload a CSV file.
                </p>
                <p style="color:rgba(255,255,255,0.35); font-size:0.85rem; margin-top:2rem;">
                    💡 <strong>Quick Example:</strong><br>
                    X: 1, 2, 3, 4, 5, 6, 7<br>
                    Y: 2.1, 4.0, 5.8, 8.2, 10.1, 12.3, 13.9
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Length Mismatch Check ──
    if len(x_data) != len(y_data):
        st.markdown(
            f'<div class="error-card" style="font-size:1.05rem;">'
            f"⚠️ <strong>Length Mismatch:</strong> "
            f"X has {len(x_data)} values but Y has {len(y_data)} values. "
            f"Both must have the same number of data points."
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Minimum Data Points Check ──
    min_points = (poly_degree + 1) if model_type == "Polynomial Regression" else 2
    if len(x_data) < min_points:
        st.markdown(
            f'<div class="error-card" style="font-size:1.05rem;">'
            f"⚠️ <strong>Insufficient Data:</strong> "
            f"Need at least {min_points} data points for "
            f"{'degree-' + str(poly_degree) + ' polynomial' if model_type == 'Polynomial Regression' else 'linear'} "
            f"regression. You provided {len(x_data)}."
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # ══════════════════════════════════════════════
    # FIT MODEL & GENERATE RESULTS
    # ══════════════════════════════════════════════
    try:
        coeffs, poly_fn, x_smooth, y_smooth = fit_model(
            x_data, y_data, model_type, poly_degree
        )
    except Exception as e:
        st.markdown(
            f'<div class="error-card">⚠️ Fitting error: {e}</div>',
            unsafe_allow_html=True,
        )
        return

    # Determine effective degree (1 for linear)
    effective_degree = 1 if model_type == "Linear Regression" else poly_degree
    equation_str = format_equation(coeffs, effective_degree)

    # Prediction
    pred_y = predict_value(poly_fn, predict_x)

    # ── Layout: Two Columns ──
    col_graph, col_info = st.columns([3, 1], gap="large")

    with col_graph:
        # ── Interactive Plotly Chart ──
        fig = create_plot(
            x_data, y_data,
            x_smooth, y_smooth,
            equation_str, model_type,
            pred_x=predict_x, pred_y=pred_y,
        )
        st.plotly_chart(fig, use_container_width=True, key="main_chart")

        # ── Equation Display ──
        st.markdown(
            f'<div class="equation-box">📐 {equation_str}</div>',
            unsafe_allow_html=True,
        )

    with col_info:
        # ── Metrics ──
        st.markdown("#### 📊 Summary")

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Points</div>
                <div class="metric-value">{len(x_data)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_label = model_type
        if model_type == "Polynomial Regression":
            model_label += f" (deg {poly_degree})"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value" style="font-size:1rem;">{model_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── R² Score (goodness of fit) ──
        y_pred_at_data = poly_fn(x_data)
        ss_res = np.sum((y_data - y_pred_at_data) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{r_squared:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Prediction Result ──
        st.markdown("#### 🔮 Prediction")
        st.markdown(
            f"""
            <div class="prediction-result">
                <div class="pred-label">f({predict_x:.4f})</div>
                <div class="pred-value">{pred_y:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Download Graph ──
        st.markdown("#### 💾 Export")
        # Plotly allows native download via the modebar, but we also
        # provide a convenient HTML download button.
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="📥 Download Graph (HTML)",
            data=html_bytes,
            file_name="trend_prediction_chart.html",
            mime="text/html",
        )

        # Also offer PNG via Plotly's static image export if kaleido is available
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            st.download_button(
                label="🖼️ Download Graph (PNG)",
                data=png_bytes,
                file_name="trend_prediction_chart.png",
                mime="image/png",
            )
        except Exception:
            st.caption(
                "💡 Install `kaleido` (`pip install kaleido`) to enable PNG export."
            )

    # ── Data Table (collapsible) ──
    with st.expander("📋 View Raw Data", expanded=False):
        data_df = pd.DataFrame({"X": x_data, "Y": y_data})
        data_df["Fitted Y"] = poly_fn(x_data)
        data_df["Residual"] = y_data - poly_fn(x_data)
        st.dataframe(
            data_df.style.format("{:.4f}").set_properties(
                **{"text-align": "center"}
            ),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
