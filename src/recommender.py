from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {
    "dataset_name",
    "description",
    "task_type",
    "domain",
    "tags",
    "source",
    "url",
    "difficulty",
}


def load_dataset_metadata(csv_path: str | Path) -> pd.DataFrame:
    """
    데이터셋 메타데이터 CSV를 읽고 필요한 컬럼을 검사한다.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    dataframe = dataframe.fillna("")

    return dataframe


def build_search_text(dataframe: pd.DataFrame) -> pd.Series:
    """
    데이터셋의 여러 메타데이터 컬럼을 하나의 검색 문장으로 합친다.

    YouTube에서 제목, 설명, 태그를 함께 사용하는 것과 같은 역할이다.
    """
    return (
        dataframe["dataset_name"].astype(str)
        + " "
        + dataframe["description"].astype(str)
        + " "
        + dataframe["task_type"].astype(str)
        + " "
        + dataframe["domain"].astype(str)
        + " "
        + dataframe["tags"].astype(str)
        + " "
        + dataframe["difficulty"].astype(str)
    )


def recommend_datasets(
    project_description: str,
    dataframe: pd.DataFrame,
    task_type: str = "All",
    domain: str = "All",
    difficulty: str = "All",
    top_n: int = 3,
) -> pd.DataFrame:
    """
    사용자 프로젝트 설명과 데이터셋 메타데이터의 코사인 유사도를 계산해
    가장 유사한 데이터셋을 반환한다.
    """
    query = project_description.strip()

    if not query:
        return pd.DataFrame()

    candidates = dataframe.copy()

    # 1. 명시적인 조건 필터링
    if task_type != "All":
        candidates = candidates[
            candidates["task_type"].str.casefold() == task_type.casefold()
        ]

    if domain != "All":
        candidates = candidates[
            candidates["domain"].str.casefold() == domain.casefold()
        ]

    if difficulty != "All":
        candidates = candidates[
            candidates["difficulty"].str.casefold() == difficulty.casefold()
        ]

    if candidates.empty:
        return pd.DataFrame()

    # 2. 데이터셋 제목·설명·태그를 하나의 문장으로 결합
    dataset_documents = build_search_text(candidates).tolist()

    # 사용자 입력도 필터 조건을 포함해 검색 문장으로 만든다.
    query_parts = [query]

    if task_type != "All":
        query_parts.append(task_type)

    if domain != "All":
        query_parts.append(domain)

    if difficulty != "All":
        query_parts.append(difficulty)

    query_document = " ".join(query_parts)

    # 3. 데이터셋 문서와 사용자 입력을 함께 TF-IDF 벡터로 변환
    all_documents = dataset_documents + [query_document]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(all_documents)

    dataset_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]

    # 4. 사용자 입력과 각 데이터셋 간의 코사인 유사도 계산
    similarity_scores = cosine_similarity(
        query_vector,
        dataset_vectors,
    ).flatten()

    # 5. 점수가 높은 순서로 정렬
    results = candidates.copy()
    results["similarity_score"] = similarity_scores

    results = results.sort_values(
        by="similarity_score",
        ascending=False,
    ).head(top_n)

    results["similarity_percent"] = (
        results["similarity_score"] * 100
    ).round(1)

    return results.reset_index(drop=True)