from __future__ import annotations

import streamlit as st

from src.live_recommender import LiveRecommender


st.set_page_config(
    page_title="AI Dataset Recommendation System",
    page_icon="📊",
    layout="wide",
)


# -------------------------------------------------
# Session state
# -------------------------------------------------
DEFAULT_STATE = {
    "recommendations": [],
    "project_description": "",
    "show_comparison": False,
}

for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
@st.cache_resource
def create_recommender() -> LiveRecommender:
    return LiveRecommender()


def value_of(dataset, name: str, default=None):
    return getattr(dataset, name, default)


def safe_text(value, default: str = "Unknown") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text or default


def safe_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return min(max(score, 0.0), 1.0)


def score_text(dataset, field: str) -> str:
    return f"{safe_score(value_of(dataset, field, 0.0)):.1%}"


def format_number(value) -> str:
    if value is None:
        return "Unknown"

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return safe_text(value)


def dataset_name(
    dataset,
    default: str = "Unnamed dataset",
) -> str:
    return safe_text(
        value_of(dataset, "dataset_name"),
        default,
    )


def dataset_key(
    dataset,
    index: int,
) -> str:
    source = safe_text(
        value_of(dataset, "source"),
        "unknown",
    )

    source_id = safe_text(
        value_of(dataset, "source_id"),
        "",
    )

    return (
        f"{source}:"
        f"{source_id or dataset_name(dataset)}:"
        f"{index}"
    )


def list_value(
    dataset,
    field: str,
) -> list:
    value = value_of(
        dataset,
        field,
        [],
    )

    return value if isinstance(value, list) else []


def reset_comparison() -> None:
    st.session_state.show_comparison = False


# -------------------------------------------------
# Shared UI blocks
# -------------------------------------------------
def show_score_metrics(dataset) -> None:
    metrics = [
        (
            "Relevance",
            "relevance_score",
        ),
        (
            "Search",
            "retrieval_score",
        ),
        (
            "Task Match",
            "task_match_score",
        ),
        (
            "Domain Match",
            "domain_match_score",
        ),
        (
            "Popularity",
            "popularity",
        ),
    ]

    columns = st.columns(
        len(metrics)
    )

    for column, (
        label,
        field,
    ) in zip(
        columns,
        metrics,
    ):
        column.metric(
            label,
            score_text(
                dataset,
                field,
            ),
        )


def show_bullet_list(
    title: str,
    items: list,
    empty_message: str = "Unknown",
) -> None:
    st.markdown(
        f"### {title}"
    )

    if items:
        for item in items:
            st.write(
                f"• {item}"
            )
    else:
        st.caption(
            empty_message
        )


def show_dataset_info(dataset) -> None:
    left, right = st.columns(2)

    left.write(
        "**Source:** "
        f"{safe_text(value_of(dataset, 'source'))}"
    )

    right.write(
        "**Data format:** "
        f"{safe_text(value_of(dataset, 'data_format'))}"
    )

    left.write(
        "**Task type:** "
        f"{safe_text(value_of(dataset, 'task_type'))}"
    )

    right.write(
        "**Domain:** "
        f"{safe_text(value_of(dataset, 'domain'))}"
    )

    left.write(
        "**Instances:** "
        f"{format_number(value_of(dataset, 'num_instances'))}"
    )

    right.write(
        "**Features:** "
        f"{format_number(value_of(dataset, 'num_features'))}"
    )

    # -------------------------------------------------
# Recommendation card
# -------------------------------------------------
def display_recommendation(
    rank: int,
    dataset,
) -> None:

    badge = {
        1: "🥇",
        2: "🥈",
        3: "🥉",
    }.get(rank, f"#{rank}")

    name = dataset_name(dataset)

    source = safe_text(
        value_of(dataset, "source")
    )

    task_type = safe_text(
        value_of(dataset, "task_type")
    )

    domain = safe_text(
        value_of(dataset, "domain")
    )

    description = safe_text(
        value_of(
            dataset,
            "description",
        ),
        "No description is available.",
    )

    final_score = safe_score(
        value_of(
            dataset,
            "final_score",
            0.0,
        )
    )

    reasons = list_value(
        dataset,
        "recommendation_reasons",
    )

    algorithms = list_value(
        dataset,
        "recommended_algorithms",
    )

    metrics = list_value(
        dataset,
        "recommended_metrics",
    )

    signals = list_value(
        dataset,
        "analysis_signals",
    )

    warnings = list_value(
        dataset,
        "warnings",
    )

    with st.container(border=True):

        title_col, score_col = st.columns(
            [4, 1],
            vertical_alignment="center",
        )

        with title_col:

            st.subheader(
                f"{badge} {name}"
            )

            st.caption(
                f"{source} · "
                f"{task_type} · "
                f"{domain}"
            )

        with score_col:

            st.metric(
                "Final Score",
                f"{final_score:.1%}",
            )

        st.progress(final_score)

        preview = description[:240]

        if len(description) > 240:
            preview += "..."

        st.write(preview)

        if reasons:
            st.info(
                f"**Key reason:** {reasons[0]}"
            )

        with st.expander(
            "View details"
        ):

            show_score_metrics(dataset)

            show_bullet_list(
                "Why recommended",
                reasons,
                "No recommendation explanation is available.",
            )

            left, right = st.columns(2)

            with left:
                show_bullet_list(
                    "Recommended algorithms",
                    algorithms,
                )

            with right:
                show_bullet_list(
                    "Recommended metrics",
                    metrics,
                )

            st.markdown(
                "### Dataset information"
            )

            show_dataset_info(dataset)

            if signals:

                show_bullet_list(
                    "Detected characteristics",
                    signals,
                )

            if warnings:

                st.markdown(
                    "### Warnings"
                )

                for warning in warnings:
                    st.warning(warning)

            url = value_of(
                dataset,
                "url",
            )

            if url:
                st.link_button(
                    "Open original dataset",
                    url,
                    use_container_width=True,
                )


# -------------------------------------------------
# Comparison
# -------------------------------------------------
def comparison_rows():

    return [

        (
            "Final Score",
            "score",
            "final_score",
        ),

        (
            "Relevance",
            "score",
            "relevance_score",
        ),

        (
            "Search Score",
            "score",
            "retrieval_score",
        ),

        (
            "Task Match",
            "score",
            "task_match_score",
        ),

        (
            "Domain Match",
            "score",
            "domain_match_score",
        ),

        (
            "Popularity",
            "score",
            "popularity",
        ),

        (
            "Task Confidence",
            "score",
            "task_confidence",
        ),

        (
            "Source",
            "text",
            "source",
        ),

        (
            "Task Type",
            "text",
            "task_type",
        ),

        (
            "Domain",
            "text",
            "domain",
        ),

        (
            "Data Format",
            "text",
            "data_format",
        ),

        (
            "Instances",
            "number",
            "num_instances",
        ),

        (
            "Features",
            "number",
            "num_features",
        ),
    ]


def comparison_value(
    dataset,
    value_type,
    field,
):

    value = value_of(
        dataset,
        field,
    )

    if value_type == "score":
        return score_text(
            dataset,
            field,
        )

    if value_type == "number":
        return format_number(value)

    return safe_text(value)

def display_comparison(
    datasets: list,
) -> None:

    if len(datasets) != 2:
        return

    st.divider()

    st.header(
        "Recommendation Result Comparison"
    )

    st.caption(
        "The table compares the recommendation "
        "engine's scores and the reasons behind "
        "each recommendation."
    )

    summary_columns = st.columns(2)

    for column, dataset in zip(
        summary_columns,
        datasets,
    ):
        with column:

            with st.container(
                border=True
            ):

                st.subheader(
                    dataset_name(dataset)
                )

                score = safe_score(
                    value_of(
                        dataset,
                        "final_score",
                        0.0,
                    )
                )

                st.metric(
                    "Final Score",
                    f"{score:.1%}",
                )

                st.progress(score)

                st.caption(
                    f"{safe_text(value_of(dataset, 'source'))}"
                    " · "
                    f"{safe_text(value_of(dataset, 'task_type'))}"
                    " · "
                    f"{safe_text(value_of(dataset, 'domain'))}"
                )

    rows = comparison_rows()

    comparison_table = {
        "Comparison Item": [
            label
            for label, _, _ in rows
        ]
    }

    for index, dataset in enumerate(
        datasets,
        start=1,
    ):

        column_name = dataset_name(
            dataset,
            f"Dataset {index}",
        )

        if column_name in comparison_table:
            column_name = (
                f"{column_name} ({index})"
            )

        comparison_table[
            column_name
        ] = [
            comparison_value(
                dataset,
                value_type,
                field,
            )
            for (
                _,
                value_type,
                field,
            ) in rows
        ]

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Recommendation Reasons"
    )

    reason_columns = st.columns(2)

    for column, dataset in zip(
        reason_columns,
        datasets,
    ):
        with column:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"#### {dataset_name(dataset)}"
                )

                reasons = list_value(
                    dataset,
                    "recommendation_reasons",
                )

                if reasons:
                    for reason in reasons:
                        st.write(
                            f"• {reason}"
                        )
                else:
                    st.caption(
                        "No recommendation reasons "
                        "are available."
                    )

                algorithms = list_value(
                    dataset,
                    "recommended_algorithms",
                )

                metrics = list_value(
                    dataset,
                    "recommended_metrics",
                )

                st.markdown(
                    "**Recommended algorithms**"
                )

                if algorithms:
                    st.write(
                        ", ".join(algorithms)
                    )
                else:
                    st.write("Unknown")

                st.markdown(
                    "**Recommended metrics**"
                )

                if metrics:
                    st.write(
                        ", ".join(metrics)
                    )
                else:
                    st.write("Unknown")

    best_dataset = max(
        datasets,
        key=lambda dataset: safe_score(
            value_of(
                dataset,
                "final_score",
                0.0,
            )
        ),
    )

    best_score = safe_score(
        value_of(
            best_dataset,
            "final_score",
            0.0,
        )
    )

    st.success(
        f"**Higher-ranked dataset: "
        f"{dataset_name(best_dataset)}**\n\n"
        f"Final recommendation score: "
        f"{best_score:.1%}"
    )


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:

    st.header(
        "Recommendation Settings"
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5,
    )

    limit_per_source = st.slider(
        "Search results per source",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
    )


# -------------------------------------------------
# Main page
# -------------------------------------------------
st.title(
    "📊 AI Dataset Recommendation System"
)

st.write(
    "Describe your AI project. "
    "The system searches UCI and Kaggle, "
    "analyzes dataset metadata, and ranks "
    "suitable datasets."
)

project_description = st.text_area(
    "Describe your AI project",
    value=(
        st.session_state.project_description
    ),
    placeholder=(
        "Example: predict house prices "
        "using income, location, and "
        "building information"
    ),
    height=180,
)

recommend_button = st.button(
    "Recommend Datasets",
    type="primary",
    use_container_width=True,
)

if recommend_button:

    query = project_description.strip()

    if not query:

        st.warning(
            "Please enter a project description."
        )

    else:

        st.session_state[
            "project_description"
        ] = query

        st.session_state[
            "show_comparison"
        ] = False

        st.session_state.pop(
            "comparison_widget",
            None,
        )

        with st.spinner(
            "Searching UCI and Kaggle "
            "datasets..."
        ):

            try:

                recommender = (
                    create_recommender()
                )

                recommendations = (
                    recommender.recommend(
                        project_description=query,
                        limit=top_n,
                        limit_per_source=(
                            limit_per_source
                        ),
                    )
                )

                st.session_state[
                    "recommendations"
                ] = recommendations

                if not recommendations:

                    st.warning(
                        "No suitable datasets "
                        "were found."
                    )

            except Exception as error:

                st.session_state[
                    "recommendations"
                ] = []

                st.error(
                    "Recommendation failed: "
                    f"{error}"
                )

# -------------------------------------------------
# Recommendation Results
# -------------------------------------------------
recommendations = st.session_state.get(
    "recommendations",
    [],
)

if recommendations:

    st.divider()

    st.header(
        "Recommended Datasets"
    )

    comparison_options = {
        dataset_name(dataset): dataset
        for dataset in recommendations
    }

    selected_names = st.multiselect(
        "Select two datasets to compare",
        options=list(
            comparison_options.keys()
        ),
        max_selections=2,
        key="comparison_widget",
    )

    compare_col, clear_col = st.columns(2)

    with compare_col:

        if st.button(
            "Compare Selected Datasets",
            use_container_width=True,
        ):

            if len(selected_names) != 2:

                st.warning(
                    "Please select exactly "
                    "two datasets."
                )

            else:

                st.session_state[
                    "show_comparison"
                ] = True

    with clear_col:

        if st.button(
            "Clear Selection",
            use_container_width=True,
        ):

            st.session_state.pop(
                "comparison_widget",
                None,
            )

            st.session_state[
                "show_comparison"
            ] = False

            st.rerun()

    for rank, dataset in enumerate(
        recommendations,
        start=1,
    ):

        display_recommendation(
            rank,
            dataset,
        )

    if (
        st.session_state.get(
            "show_comparison",
            False,
        )
        and len(selected_names) == 2
    ):

        selected_datasets = [
            comparison_options[name]
            for name in selected_names
        ]

        display_comparison(
            selected_datasets
        )

else:

    st.info(
        "Enter your project description "
        "and press **Recommend Datasets**."
    )