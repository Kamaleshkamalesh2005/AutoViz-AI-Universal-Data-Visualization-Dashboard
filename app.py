from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="Automatic Data Visualization Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


PLOTLY_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#f4f7fb"},
    "margin": {"l": 20, "r": 20, "t": 55, "b": 20},
}


def apply_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #08111f;
                --panel: rgba(12, 22, 39, 0.92);
                --panel-strong: rgba(17, 29, 49, 0.96);
                --text: #eef3ff;
                --muted: #91a2be;
                --border: rgba(132, 162, 212, 0.18);
                --cyan: #32d4ff;
                --green: #00d084;
                --gold: #ffb84d;
                --pink: #ff6b9f;
            }

            .stApp {
                background:
                    radial-gradient(circle at 15% 15%, rgba(50, 212, 255, 0.16), transparent 24%),
                    radial-gradient(circle at 85% 8%, rgba(255, 107, 159, 0.12), transparent 22%),
                    radial-gradient(circle at 50% 100%, rgba(0, 208, 132, 0.12), transparent 24%),
                    linear-gradient(180deg, #09111d 0%, #040912 100%);
                color: var(--text);
                font-family: "Segoe UI", "Aptos", sans-serif;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(9, 18, 34, 0.98), rgba(6, 12, 24, 0.98));
                border-right: 1px solid var(--border);
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(20, 32, 54, 0.96), rgba(11, 20, 36, 0.92));
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 1.45rem 1.6rem;
                box-shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
                margin-bottom: 1.1rem;
                position: relative;
                overflow: hidden;
            }

            .hero-card::after {
                content: "";
                position: absolute;
                inset: auto -10% -35% auto;
                width: 260px;
                height: 260px;
                background: radial-gradient(circle, rgba(50, 212, 255, 0.2), transparent 70%);
            }

            .hero-title {
                font-size: 2rem;
                font-weight: 800;
                margin: 0;
                letter-spacing: -0.03em;
            }

            .hero-subtitle {
                color: var(--muted);
                margin-top: 0.45rem;
                font-size: 1rem;
                max-width: 900px;
            }

            .section-title {
                font-size: 1.05rem;
                font-weight: 800;
                color: var(--text);
                margin: 1rem 0 0.65rem 0;
                letter-spacing: 0.02em;
            }

            .metric-shell {
                background: linear-gradient(180deg, rgba(18, 29, 49, 0.95), rgba(10, 18, 32, 0.95));
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1rem 1rem 0.85rem 1rem;
                box-shadow: 0 18px 32px rgba(0, 0, 0, 0.22);
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .metric-value {
                color: var(--text);
                font-size: 1.85rem;
                font-weight: 800;
                line-height: 1.1;
                margin-top: 0.3rem;
            }

            .metric-accent {
                width: 54px;
                height: 4px;
                border-radius: 999px;
                margin-top: 0.65rem;
            }

            .card-shell {
                background: linear-gradient(180deg, rgba(14, 24, 42, 0.95), rgba(9, 17, 31, 0.95));
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1rem 1rem 0.8rem 1rem;
                box-shadow: 0 16px 34px rgba(0, 0, 0, 0.22);
                min-height: 100%;
            }

            .card-title {
                color: var(--text);
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }

            .card-subtitle {
                color: var(--muted);
                font-size: 0.86rem;
                margin-bottom: 0.6rem;
            }

            .schema-chip {
                display: inline-block;
                padding: 0.28rem 0.65rem;
                margin: 0.15rem 0.35rem 0.15rem 0;
                background: rgba(50, 212, 255, 0.12);
                border: 1px solid rgba(50, 212, 255, 0.18);
                border-radius: 999px;
                color: #c7efff;
                font-size: 0.82rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def sample_dataframe(dataframe: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    if len(dataframe) <= max_rows:
        return dataframe.copy()
    return dataframe.sample(max_rows, random_state=42).copy()


def detect_column_types(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    working_df = dataframe.copy()
    datetime_columns: set[str] = set()

    object_like_columns = working_df.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in object_like_columns:
        non_null = working_df[column].dropna()
        if non_null.empty:
            continue

        sample_values = non_null.astype(str).head(400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample_values, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                converted = pd.to_datetime(working_df[column], errors="coerce")
            if converted.notna().sum() > 0:
                working_df[column] = converted
                datetime_columns.add(column)

    datetime_columns.update(working_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist())
    numeric_columns = working_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column
        for column in working_df.columns
        if column not in numeric_columns and column not in datetime_columns
    ]
    return working_df, numeric_columns, categorical_columns, sorted(datetime_columns)


def build_summary(dataframe: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    try:
        summary = dataframe.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        summary = dataframe.describe(include="all").transpose()

    if numeric_columns:
        variance = dataframe[numeric_columns].var(numeric_only=True).sort_values(ascending=False)
        summary["variance"] = variance.reindex(summary.index)
    return summary.fillna("-")


def render_metric_card(label: str, value: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-shell">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-accent" style="background: {accent};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def start_card(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="card-shell">
            <div class="card-title">{title}</div>
            <div class="card-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def end_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def finalize_plotly_figure(fig, height: int = 360):
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    return fig


def render_dataframe(dataframe: pd.DataFrame, hide_index: bool = False) -> None:
    try:
        st.dataframe(dataframe, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(dataframe, use_container_width=True, hide_index=hide_index)


def render_plotly(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def generate_histograms(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list:
    figures = []
    sampled = sample_dataframe(dataframe[numeric_columns], max_rows=4000) if numeric_columns else pd.DataFrame()
    palette = ["#32d4ff", "#8a7dff", "#ff6b9f", "#00d084"]
    for index, column in enumerate(numeric_columns[:4]):
        fig = px.histogram(sampled, x=column, nbins=30, color_discrete_sequence=[palette[index % len(palette)]])
        figures.append(finalize_plotly_figure(fig, height=340))
    return figures


def generate_kde_plots(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    if not numeric_columns:
        return figures

    for column in numeric_columns[:3]:
        plot_df = sample_dataframe(dataframe[[column]].dropna(), max_rows=1500)
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sns.kdeplot(plot_df[column], fill=True, color="#32d4ff", linewidth=2, ax=ax)
        ax.set_facecolor("#101a2d")
        fig.patch.set_facecolor("#101a2d")
        ax.tick_params(colors="#dce7ff")
        ax.set_title(f"KDE Plot: {column}", color="#eef3ff")
        ax.set_xlabel(column, color="#dce7ff")
        ax.set_ylabel("Density", color="#dce7ff")
        for spine in ax.spines.values():
            spine.set_color("#304562")
        figures.append(fig)
    return figures


def generate_bar_charts(dataframe: pd.DataFrame, categorical_columns: list[str]) -> list:
    figures = []
    for column in categorical_columns[:4]:
        counts = dataframe[column].fillna("Missing").astype(str).value_counts().head(12).reset_index()
        counts.columns = [column, "Count"]
        fig = px.bar(
            counts,
            x=column,
            y="Count",
            color="Count",
            color_continuous_scale=["#113a5d", "#32d4ff", "#00d084"],
        )
        fig.update_layout(coloraxis_showscale=False)
        figures.append(finalize_plotly_figure(fig, height=340))
    return figures


def generate_pie_charts(dataframe: pd.DataFrame, categorical_columns: list[str]) -> list:
    figures = []
    for column in categorical_columns[:3]:
        counts = dataframe[column].fillna("Missing").astype(str).value_counts().head(8).reset_index()
        counts.columns = [column, "Count"]
        fig = px.pie(
            counts,
            names=column,
            values="Count",
            hole=0.5,
            color_discrete_sequence=["#32d4ff", "#00d084", "#ffb84d", "#ff6b9f", "#8a7dff", "#5dd6c0"],
        )
        figures.append(finalize_plotly_figure(fig, height=340))
    return figures


def generate_count_plots(dataframe: pd.DataFrame, categorical_columns: list[str]) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    for column in categorical_columns[:3]:
        counts = dataframe[column].fillna("Missing").astype(str)
        ordered = counts.value_counts().head(10).index.tolist()
        if not ordered:
            continue
        plot_df = counts[counts.isin(ordered)].to_frame(name=column)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sns.countplot(data=plot_df, x=column, order=ordered, hue=column, palette="crest", legend=False, ax=ax)
        fig.patch.set_facecolor("#101a2d")
        ax.set_facecolor("#101a2d")
        ax.tick_params(colors="#dce7ff", rotation=30)
        ax.set_title(f"Count Plot: {column}", color="#eef3ff")
        ax.set_xlabel(column, color="#dce7ff")
        ax.set_ylabel("Count", color="#dce7ff")
        for spine in ax.spines.values():
            spine.set_color("#304562")
        figures.append(fig)
    return figures


def generate_scatter_plots(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list:
    figures = []
    if len(numeric_columns) < 2:
        return figures

    sampled = sample_dataframe(dataframe[numeric_columns], max_rows=2500)
    ranked = dataframe[numeric_columns].var(numeric_only=True).sort_values(ascending=False).index.tolist()
    selected = ranked[: min(4, len(ranked))]
    for index in range(len(selected) - 1):
        x_axis = selected[index]
        y_axis = selected[index + 1]
        fig = px.scatter(
            sampled,
            x=x_axis,
            y=y_axis,
            color=y_axis,
            color_continuous_scale=["#16324f", "#32d4ff", "#ff6b9f"],
            opacity=0.8,
        )
        fig.update_layout(coloraxis_showscale=False)
        figures.append(finalize_plotly_figure(fig, height=340))
    return figures


def generate_line_charts(dataframe: pd.DataFrame, numeric_columns: list[str], datetime_columns: list[str]) -> list:
    figures = []
    if datetime_columns and numeric_columns:
        time_column = datetime_columns[0]
        line_columns = numeric_columns[: min(3, len(numeric_columns))]
        plot_df = dataframe[[time_column] + line_columns].dropna(subset=[time_column]).sort_values(time_column)
        plot_df = sample_dataframe(plot_df, max_rows=3000)
        melted = plot_df.melt(id_vars=time_column, value_vars=line_columns, var_name="Metric", value_name="Value")
        figures.append(finalize_plotly_figure(px.line(melted, x=time_column, y="Value", color="Metric"), height=340))
        return figures

    if len(numeric_columns) >= 2:
        plot_df = sample_dataframe(dataframe[numeric_columns[:2]].reset_index(), max_rows=2500)
        figures.append(finalize_plotly_figure(px.line(plot_df, x="index", y=numeric_columns[:2]), height=340))
    return figures


def generate_box_plots(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list:
    figures = []
    sampled = sample_dataframe(dataframe[numeric_columns], max_rows=3500) if numeric_columns else pd.DataFrame()
    for column in numeric_columns[:4]:
        fig = px.box(sampled, y=column, color_discrete_sequence=["#ffb84d"])
        figures.append(finalize_plotly_figure(fig, height=340))
    return figures


def generate_heatmap(dataframe: pd.DataFrame, numeric_columns: list[str]) -> plt.Figure | None:
    if len(numeric_columns) < 2:
        return None
    correlation = dataframe[numeric_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.heatmap(correlation, cmap="mako", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    fig.patch.set_facecolor("#101a2d")
    ax.set_facecolor("#101a2d")
    ax.tick_params(colors="#dce7ff", rotation=30)
    ax.set_title("Correlation Heatmap", color="#eef3ff")
    return fig


def generate_missing_values_chart(dataframe: pd.DataFrame):
    missing = dataframe.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return None
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing Values",
        color="Missing Values",
        color_continuous_scale=["#40203c", "#ff6b9f", "#ffb84d"],
    )
    fig.update_layout(coloraxis_showscale=False)
    return finalize_plotly_figure(fig, height=320)


def generate_missing_heatmap(dataframe: pd.DataFrame) -> plt.Figure:
    sampled = sample_dataframe(dataframe, max_rows=500)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.heatmap(sampled.isna(), cmap="rocket_r", cbar=False, yticklabels=False, ax=ax)
    fig.patch.set_facecolor("#101a2d")
    ax.set_facecolor("#101a2d")
    ax.set_title("Missing Data Heatmap", color="#eef3ff")
    ax.tick_params(colors="#dce7ff")
    return fig


def generate_dataset_insights(dataframe: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if len(numeric_columns) >= 2:
        correlation = dataframe[numeric_columns].corr(numeric_only=True)
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        top_corr = upper.stack().abs().sort_values(ascending=False).head(5).reset_index()
        if not top_corr.empty:
            top_corr.columns = ["Feature 1", "Feature 2", "Absolute Correlation"]
        else:
            top_corr = pd.DataFrame(columns=["Feature 1", "Feature 2", "Absolute Correlation"])
    else:
        top_corr = pd.DataFrame(columns=["Feature 1", "Feature 2", "Absolute Correlation"])

    if numeric_columns:
        top_variance = dataframe[numeric_columns].var(numeric_only=True).sort_values(ascending=False).head(5).reset_index()
        top_variance.columns = ["Column", "Variance"]
    else:
        top_variance = pd.DataFrame(columns=["Column", "Variance"])

    summary_text = (
        f"The dataset contains {len(dataframe):,} rows and {len(dataframe.columns):,} columns, "
        f"with {len(numeric_columns):,} numeric features and {len(categorical_columns):,} categorical features. "
        f"It includes {int(dataframe.isna().sum().sum()):,} missing values and {int(dataframe.duplicated().sum()):,} duplicate rows."
    )
    return top_corr, top_variance, summary_text


def render_custom_chart(dataframe: pd.DataFrame, chart_type: str, x_axis: str, y_axis: str | None):
    plot_df = sample_dataframe(dataframe, max_rows=4000)
    chart_key = chart_type.lower()
    if chart_key == "scatter":
        if not y_axis:
            return None, "Select both X and Y axes for a scatter chart."
        fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=y_axis, color_continuous_scale="tealrose")
    elif chart_key == "line":
        if not y_axis:
            return None, "Select both X and Y axes for a line chart."
        fig = px.line(plot_df.sort_values(x_axis), x=x_axis, y=y_axis)
    elif chart_key == "bar":
        if y_axis:
            fig = px.bar(plot_df, x=x_axis, y=y_axis, color=y_axis, color_continuous_scale="viridis")
        else:
            counts = plot_df[x_axis].fillna("Missing").astype(str).value_counts().head(12).reset_index()
            counts.columns = [x_axis, "Count"]
            fig = px.bar(counts, x=x_axis, y="Count", color="Count", color_continuous_scale="viridis")
    elif chart_key == "histogram":
        fig = px.histogram(plot_df, x=x_axis, nbins=30, color_discrete_sequence=["#32d4ff"])
    elif chart_key == "pie":
        if y_axis:
            pie_df = plot_df[[x_axis, y_axis]].dropna()
            fig = px.pie(pie_df, names=x_axis, values=y_axis)
        else:
            counts = plot_df[x_axis].fillna("Missing").astype(str).value_counts().head(10).reset_index()
            counts.columns = [x_axis, "Count"]
            fig = px.pie(counts, names=x_axis, values="Count")
    elif chart_key == "box":
        fig = px.box(plot_df, x=x_axis if y_axis else None, y=y_axis or x_axis, color_discrete_sequence=["#ffb84d"])
    else:
        return None, "Unsupported chart type selected."

    return finalize_plotly_figure(fig, height=420), None


def load_default_sample() -> bytes | None:
    sample_path = Path(__file__).with_name("global air pollution dataset.csv")
    if sample_path.exists():
        return sample_path.read_bytes()
    return None


def render_plot_card(title: str, subtitle: str, fig=None, mpl_fig: plt.Figure | None = None, empty_message: str = "No chart available.") -> None:
    start_card(title, subtitle)
    if fig is not None:
        render_plotly(fig)
    elif mpl_fig is not None:
        st.pyplot(mpl_fig, clear_figure=True)
        plt.close(mpl_fig)
    else:
        st.info(empty_message)
    end_card()


def main() -> None:
    apply_dashboard_theme()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Automatic Data Visualization Dashboard</div>
            <div class="hero-subtitle">Upload any dataset and generate instant analytics with a dark business intelligence style dashboard, responsive grid charts, and interactive exploration controls.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Dataset Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
    sample_bytes = load_default_sample()
    use_sample = st.sidebar.checkbox("Use bundled sample dataset", value=False, disabled=sample_bytes is None)

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_name = uploaded_file.name
    elif use_sample and sample_bytes is not None:
        file_bytes = sample_bytes
        source_name = "global air pollution dataset.csv"
    else:
        file_bytes = None
        source_name = None

    if not file_bytes:
        st.info("Upload a CSV file from the sidebar, or enable the bundled sample dataset to generate the dashboard.")
        return

    try:
        raw_df = load_dataset(file_bytes)
    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        return

    dataframe, numeric_columns, categorical_columns, datetime_columns = detect_column_types(raw_df)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Information")
    st.sidebar.write(f"Source: {source_name}")
    st.sidebar.write(f"Rows: {len(dataframe):,}")
    st.sidebar.write(f"Columns: {len(dataframe.columns):,}")
    st.sidebar.write(f"Missing values: {int(dataframe.isna().sum().sum()):,}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Column Filters")
    selected_numeric = st.sidebar.multiselect(
        "Numeric columns",
        options=numeric_columns,
        default=numeric_columns[: min(6, len(numeric_columns))],
    )
    selected_categorical = st.sidebar.multiselect(
        "Categorical columns",
        options=categorical_columns,
        default=categorical_columns[: min(4, len(categorical_columns))],
    )
    selected_datetime = st.sidebar.multiselect(
        "Datetime columns",
        options=datetime_columns,
        default=datetime_columns[:1],
    )

    if not selected_numeric:
        selected_numeric = numeric_columns[: min(6, len(numeric_columns))]
    if not selected_categorical:
        selected_categorical = categorical_columns[: min(4, len(categorical_columns))]
    if not selected_datetime:
        selected_datetime = datetime_columns[:1]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Chart Selection Options")
    chart_type = st.sidebar.selectbox("Chart type", ["scatter", "line", "bar", "histogram", "pie", "box"])
    x_axis = st.sidebar.selectbox("X-axis column", dataframe.columns.tolist(), index=0)
    y_choices = [None] + dataframe.columns.tolist()
    y_axis = st.sidebar.selectbox("Y-axis column", y_choices, index=1 if len(y_choices) > 1 else 0, format_func=lambda value: "None" if value is None else value)

    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    overview_left, overview_right = st.columns([1.4, 1])
    with overview_left:
        start_card("Preview Table", "First 10 rows from the uploaded dataset")
        render_dataframe(dataframe.head(10))
        end_card()
    with overview_right:
        start_card("Schema and Types", "Detected columns, missing values, and inferred data categories")
        type_df = pd.DataFrame(
            {
                "Column": dataframe.columns,
                "Data Type": dataframe.dtypes.astype(str).values,
                "Missing": dataframe.isna().sum().values,
            }
        )
        render_dataframe(type_df, hide_index=True)
        if selected_numeric:
            st.markdown("".join([f'<span class="schema-chip">Numeric: {column}</span>' for column in selected_numeric]), unsafe_allow_html=True)
        if selected_categorical:
            st.markdown("".join([f'<span class="schema-chip">Category: {column}</span>' for column in selected_categorical]), unsafe_allow_html=True)
        if selected_datetime:
            st.markdown("".join([f'<span class="schema-chip">Datetime: {column}</span>' for column in selected_datetime]), unsafe_allow_html=True)
        end_card()

    st.markdown('<div class="section-title">KPI Metrics</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total Rows", f"{len(dataframe):,}", "linear-gradient(90deg, #32d4ff, #8a7dff)")
    with metric_cols[1]:
        render_metric_card("Total Columns", f"{len(dataframe.columns):,}", "linear-gradient(90deg, #00d084, #32d4ff)")
    with metric_cols[2]:
        render_metric_card("Numeric Features", f"{len(numeric_columns):,}", "linear-gradient(90deg, #ffb84d, #ff6b9f)")
    with metric_cols[3]:
        render_metric_card("Categorical Features", f"{len(categorical_columns):,}", "linear-gradient(90deg, #ff6b9f, #8a7dff)")

    histograms = generate_histograms(dataframe, selected_numeric)
    scatter_plots = generate_scatter_plots(dataframe, selected_numeric)
    pie_charts = generate_pie_charts(dataframe, selected_categorical)
    bar_charts = generate_bar_charts(dataframe, selected_categorical)
    box_plots = generate_box_plots(dataframe, selected_numeric)
    line_charts = generate_line_charts(dataframe, selected_numeric, selected_datetime)
    correlation_heatmap = generate_heatmap(dataframe, selected_numeric)
    custom_chart, custom_error = render_custom_chart(dataframe, chart_type, x_axis, y_axis)
    kde_figs = generate_kde_plots(dataframe, selected_numeric)
    count_figs = generate_count_plots(dataframe, selected_categorical)

    st.markdown('<div class="section-title">Automatic Chart Generation</div>', unsafe_allow_html=True)
    row2 = st.columns(3)
    with row2[0]:
        render_plot_card("Histogram", "Distribution view for a top numeric feature", fig=histograms[0] if histograms else None)
    with row2[1]:
        render_plot_card("Scatter Plot", "Relationship between high-variance numeric features", fig=scatter_plots[0] if scatter_plots else None, empty_message="Not enough numeric columns for a scatter plot.")
    with row2[2]:
        render_plot_card("Pie Chart", "Category share for a top categorical feature", fig=pie_charts[0] if pie_charts else None, empty_message="No categorical columns available for a pie chart.")

    row3 = st.columns(3)
    with row3[0]:
        render_plot_card("Bar Chart", "Frequency distribution of a categorical column", fig=bar_charts[0] if bar_charts else None, empty_message="No categorical columns available for a bar chart.")
    with row3[1]:
        render_plot_card("Box Plot", "Spread and outlier analysis for a numeric feature", fig=box_plots[0] if box_plots else None, empty_message="No numeric columns available for a box plot.")
    with row3[2]:
        render_plot_card("Line Chart", "Trend view using datetime or index progression", fig=line_charts[0] if line_charts else None, empty_message="A line chart needs at least one numeric feature.")

    row4 = st.columns(2)
    with row4[0]:
        render_plot_card("Correlation Analysis", "Correlation heatmap across detected numeric features", mpl_fig=correlation_heatmap, empty_message="At least two numeric columns are required for a correlation heatmap.")
    with row4[1]:
        start_card("Interactive Visualization Builder", "Select X-axis, Y-axis, and chart type from the sidebar")
        if custom_error:
            st.warning(custom_error)
        elif custom_chart is not None:
            render_plotly(custom_chart)
        else:
            st.info("Custom chart could not be generated.")
        end_card()

    st.markdown('<div class="section-title">Distribution Analysis</div>', unsafe_allow_html=True)
    distribution_cols = st.columns(2)
    with distribution_cols[0]:
        render_plot_card("Additional Histogram", "Another numeric distribution panel", fig=histograms[1] if len(histograms) > 1 else None, empty_message="Not enough numeric columns for another histogram.")
    with distribution_cols[1]:
        render_plot_card("KDE Plot", "Smoothed density curve for a top numeric feature", mpl_fig=kde_figs[0] if kde_figs else None, empty_message="No numeric columns available for KDE analysis.")

    st.markdown('<div class="section-title">Category Analysis</div>', unsafe_allow_html=True)
    category_cols = st.columns(2)
    with category_cols[0]:
        render_plot_card("Category Frequency", "Top category counts for the selected fields", fig=bar_charts[1] if len(bar_charts) > 1 else (bar_charts[0] if bar_charts else None), empty_message="No categorical columns available for category analysis.")
    with category_cols[1]:
        render_plot_card("Count Plot", "Count plot for a categorical field", mpl_fig=count_figs[0] if count_figs else None, empty_message="No categorical columns available for a count plot.")

    st.markdown('<div class="section-title">Relationship Analysis</div>', unsafe_allow_html=True)
    relationship_cols = st.columns(2)
    with relationship_cols[0]:
        render_plot_card("Scatter Comparison", "Another cross-feature relationship plot", fig=scatter_plots[1] if len(scatter_plots) > 1 else None, empty_message="Not enough numeric columns for multiple relationship charts.")
    with relationship_cols[1]:
        render_plot_card("Trend Comparison", "Secondary line trend for selected measures", fig=line_charts[1] if len(line_charts) > 1 else None, empty_message="Only one line chart was available for this dataset.")

    st.markdown('<div class="section-title">Missing Value Analysis</div>', unsafe_allow_html=True)
    missing_cols = st.columns(2)
    with missing_cols[0]:
        render_plot_card("Missing Values per Column", "Columns with incomplete data", fig=generate_missing_values_chart(dataframe), empty_message="No missing values detected in the dataset.")
    with missing_cols[1]:
        render_plot_card("Missing Data Heatmap", "Row and column level view of gaps in the dataset", mpl_fig=generate_missing_heatmap(dataframe))

    st.markdown('<div class="section-title">Dataset Insights</div>', unsafe_allow_html=True)
    top_corr, top_variance, summary_text = generate_dataset_insights(dataframe, selected_numeric, selected_categorical)
    insight_cols = st.columns([1.1, 1, 1])
    with insight_cols[0]:
        start_card("Summary", "Automatic business intelligence style overview")
        st.write(summary_text)
        st.write(f"Detected datetime columns: {selected_datetime or 'None'}")
        st.write(f"Dataset shape: {dataframe.shape[0]:,} rows x {dataframe.shape[1]:,} columns")
        end_card()
    with insight_cols[1]:
        start_card("Highest Correlated Features", "Top absolute correlations among numeric features")
        render_dataframe(top_corr, hide_index=True)
        end_card()
    with insight_cols[2]:
        start_card("Highest Variance Columns", "Most variable numeric features in the dataset")
        render_dataframe(top_variance, hide_index=True)
        end_card()

    with st.expander("Detailed Statistics", expanded=False):
        render_dataframe(build_summary(dataframe, selected_numeric))


if __name__ == "__main__":
    main()