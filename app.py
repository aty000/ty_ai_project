from pathlib import Path

import streamlit as st

from src.recommender import (
    load_dataset_metadata,
    recommend_datasets,
)


BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_metadata.csv"
)


st.set_page_config(
    page_title="AI Resource Recommendation System",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data():
    """
    로컬 데이터셋 메타데이터를 캐시에 저장한다.
    """
    return load_dataset_metadata(METADATA_PATH)


def display_recommendation(
    rank: int,
    dataset,
) -> None:
    """
    추천 데이터셋 한 개를 Streamlit 카드 형태로 출력한다.
    """

    st.subheader(
        f"{rank}. {dataset['dataset_name']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Similarity",
            f"{dataset['similarity_percent']}%",
        )

    with col2:
        st.write("**Task Type**")
        st.write(dataset["task_type"])

    with col3:
        st.write("**Domain**")
        st.write(dataset["domain"])

    with col4:
        st.write("**Difficulty**")
        st.write(dataset["difficulty"])

    st.write(dataset["description"])

    st.caption(
        f"Source: {dataset['source']} | "
        f"Tags: {dataset['tags']}"
    )

    st.link_button(
        "Open original dataset",
        dataset["url"],
    )

    st.divider()


st.title("📊 AI Resource Recommendation System")

st.write(
    "Describe the goal of your AI project and receive dataset "
    "recommendations based on content similarity."
)


try:
    dataset_metadata = load_data()

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


with st.sidebar:
    st.header("Project Conditions")

    task_types = ["All"] + sorted(
        dataset_metadata["task_type"]
        .dropna()
        .unique()
        .tolist()
    )

    domains = ["All"] + sorted(
        dataset_metadata["domain"]
        .dropna()
        .unique()
        .tolist()
    )

    difficulties = ["All"] + sorted(
        dataset_metadata["difficulty"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_task_type = st.selectbox(
        "Task Type",
        task_types,
    )

    selected_domain = st.selectbox(
        "Domain",
        domains,
    )

    selected_difficulty = st.selectbox(
        "Difficulty",
        difficulties,
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=5,
        value=3,
    )


project_description = st.text_area(
    "Describe your AI project",
    placeholder=(
        "Example: I want to predict household electricity consumption "
        "using historical hourly measurements."
    ),
    height=180,
)


recommend_button = st.button(
    "Recommend Datasets",
    type="primary",
    use_container_width=True,
)


if recommend_button:
    if not project_description.strip():
        st.warning(
            "Please enter a project description."
        )

    else:
        with st.spinner(
            "Searching for relevant datasets..."
        ):
            recommendations = recommend_datasets(
                project_description=project_description,
                dataframe=dataset_metadata,
                task_type=selected_task_type,
                domain=selected_domain,
                difficulty=selected_difficulty,
                top_n=top_n,
            )

        if recommendations.empty:
            st.warning(
                "No datasets matched the selected conditions. "
                "Try changing one or more filters to All."
            )

        else:
            st.header(
                "Recommended Datasets"
            )

            result_count = len(recommendations)

            if result_count < top_n:
                st.info(
                    f"Only {result_count} dataset(s) with sufficient "
                    "relevance were found. Irrelevant results were excluded."
                )
            else:
                st.caption(
                    f"Showing {result_count} recommended dataset(s)."
                )

            for rank, (_, dataset) in enumerate(
                recommendations.iterrows(),
                start=1,
            ):
                display_recommendation(
                    rank=rank,
                    dataset=dataset,
                )