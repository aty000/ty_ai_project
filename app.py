import streamlit as st

from src.live_recommender import LiveRecommender


st.set_page_config(
    page_title="AI Dataset Recommendation System",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def create_recommender() -> LiveRecommender:
    """
    LiveRecommender 객체를 한 번만 생성하여 재사용한다.
    """
    return LiveRecommender()


def display_recommendation(
    rank: int,
    dataset,
) -> None:
    """
    추천 데이터셋 한 개를 카드 형태로 출력한다.
    """
    with st.container(border=True):
        st.subheader(
            f"{rank}. {dataset.dataset_name}"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Source",
            dataset.source,
        )

        col2.metric(
            "Relevance",
            f"{dataset.relevance_score:.3f}",
        )

        col3.metric(
            "Popularity",
            f"{dataset.popularity:.3f}",
        )

        col4.metric(
            "Final Score",
            f"{dataset.final_score:.3f}",
        )

        if dataset.description:
            st.write(dataset.description)

        detail_col1, detail_col2 = st.columns(2)

        detail_col1.write(
            f"**Task Type:** "
            f"{dataset.task_type or 'Unknown'}"
        )

        detail_col2.write(
            f"**Domain:** "
            f"{dataset.domain or 'Unknown'}"
        )

        st.write(
            f"**Task Confidence:** "
            f"{dataset.task_confidence:.0%}"
        )

        st.write(
            "**Recommended Metrics:** "
            + ", ".join(dataset.recommended_metrics)
        )

        st.write("**Analysis Signals:**")

        for signal in dataset.analysis_signals:
            st.write(f"- {signal}")

        if dataset.num_instances is not None:
            st.write(
                f"**Number of instances:** "
                f"{dataset.num_instances:,}"
            )

        if dataset.num_features is not None:
            st.write(
                f"**Number of features:** "
                f"{dataset.num_features:,}"
            )

        if dataset.url:
            st.link_button(
                "Open original dataset",
                dataset.url,
            )


st.title("📊 AI Dataset Recommendation System")

st.write(
    "Describe your AI project. The system searches UCI and Kaggle "
    "and ranks datasets using relevance and popularity."
)


with st.sidebar:
    st.header("Recommendation Settings")

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


project_description = st.text_area(
    "Describe your AI project",
    placeholder=(
        "Example: house prices using income and location data"
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
        with st.spinner(
            "Searching UCI and Kaggle datasets..."
        ):
            try:
                recommender = create_recommender()

                recommendations = recommender.recommend(
                    project_description=query,
                    limit=top_n,
                    limit_per_source=limit_per_source,
                )

            except Exception as error:
                st.error(
                    f"Recommendation failed: {error}"
                )
                recommendations = []

        if not recommendations:
            st.warning(
                "No sufficiently relevant datasets were found."
            )

        else:
            st.header("Recommended Datasets")

            st.caption(
                f"Showing {len(recommendations)} dataset(s)."
            )

            for rank, dataset in enumerate(
                recommendations,
                start=1,
            ):
                display_recommendation(
                    rank=rank,
                    dataset=dataset,
                )