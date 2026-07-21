from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity
from ucimlrepo import fetch_ucirepo

from src.collectors.base import DatasetCollector
from src.collectors.common import DatasetCandidate
from src.collectors.uci_catalog import load_uci_catalog


UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset"


def _metadata_value(
    metadata: Any,
    key: str,
    default: Any = "",
) -> Any:
    """
    metadata가 딕셔너리 또는 객체 형태인 경우를 모두 처리한다.
    """

    if metadata is None:
        return default

    if isinstance(metadata, dict):
        value = metadata.get(key, default)
    else:
        value = getattr(metadata, key, default)

    return default if value is None else value


def _normalize_text(
    value: Any,
    default: str = "Unknown",
) -> str:
    """
    문자열, 리스트, 튜플 등의 값을 하나의 문자열로 변환한다.
    """

    if value is None:
        return default

    if isinstance(value, (list, tuple, set)):
        values = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

        return ", ".join(values) if values else default

    text = str(value).strip()

    return text if text else default


def _safe_int(
    value: Any,
) -> int | None:
    """
    UCI 메타데이터의 행 수와 특성 수를 안전하게 정수로 변환한다.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class UCICollector(DatasetCollector):
    """
    UCI 데이터셋 카탈로그를 검색하는 수집기.

    카탈로그는 로컬 CSV 캐시를 사용하고,
    한 Collector 인스턴스에서는 TF-IDF 인덱스를 한 번만 생성한다.
    """

    source_name = "UCI"

    def __init__(self) -> None:
        self._catalog: pd.DataFrame | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._catalog_matrix: Any = None

    def _ensure_index(self) -> None:
        """
        카탈로그와 TF-IDF 행렬을 최초 한 번만 생성한다.
        """

        if (
            self._catalog is not None
            and self._vectorizer is not None
            and self._catalog_matrix is not None
        ):
            return

        catalog = load_uci_catalog()

        if catalog.empty:
            raise RuntimeError(
                "UCI 카탈로그가 비어 있습니다."
            )

        if "name" not in catalog.columns:
            raise RuntimeError(
                "UCI 카탈로그에 name 컬럼이 없습니다."
            )

        catalog = catalog.copy()

        catalog["name"] = (
            catalog["name"]
            .fillna("")
            .astype(str)
        )

        custom_stop_words = ENGLISH_STOP_WORDS.union(
            {
                "data",
                "dataset",
                "datasets",
                "predict",
                "prediction",
                "predicting",
                "using",
                "use",
                "want",
                "model",
                "models",
                "machine",
                "learning",
                "project",
            }
        )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(custom_stop_words),
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        catalog_matrix = vectorizer.fit_transform(
            catalog["name"]
        )

        self._catalog = catalog
        self._vectorizer = vectorizer
        self._catalog_matrix = catalog_matrix

    def _rank_catalog(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """
        사용자 검색 문장과 UCI 데이터셋 이름의 유사도를 계산한다.
        """

        self._ensure_index()

        assert self._catalog is not None
        assert self._vectorizer is not None
        assert self._catalog_matrix is not None

        query_vector = self._vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            self._catalog_matrix,
        ).flatten()

        ranked = self._catalog.copy()
        ranked["catalog_score"] = scores

        ranked = (
            ranked[ranked["catalog_score"] > 0]
            .sort_values(
                by="catalog_score",
                ascending=False,
            )
            .head(limit)
        )

        return ranked.to_dict(
            orient="records"
        )

    def _convert_item(
        self,
        item: dict[str, Any],
    ) -> DatasetCandidate | None:
        """
        UCI 카탈로그 항목을 DatasetCandidate로 변환한다.
        """

        dataset_id = item.get("id")

        if dataset_id is None:
            return None

        try:
            numeric_id = int(dataset_id)
        except (TypeError, ValueError):
            return None

        fallback_url = (
            f"{UCI_DATASET_URL}/{numeric_id}"
        )

        try:
            catalog_score = float(
                item.get("catalog_score", 0.0)
            )
        except (TypeError, ValueError):
            catalog_score = 0.0

        try:
            dataset = fetch_ucirepo(
                id=numeric_id
            )

            metadata = dataset.metadata

            return DatasetCandidate(
                dataset_name=_normalize_text(
                    _metadata_value(
                        metadata,
                        "name",
                        item.get("name"),
                    ),
                    default=f"UCI Dataset {numeric_id}",
                ),
                description=_normalize_text(
                    _metadata_value(
                        metadata,
                        "abstract",
                        "",
                    ),
                    default="No description available.",
                ),
                task_type=_normalize_text(
                    _metadata_value(
                        metadata,
                        "task",
                        "Unknown",
                    )
                ),
                domain=_normalize_text(
                    _metadata_value(
                        metadata,
                        "area",
                        "Unknown",
                    )
                ),
                source=self.source_name,
                url=_normalize_text(
                    _metadata_value(
                        metadata,
                        "repository_url",
                        fallback_url,
                    ),
                    default=fallback_url,
                ),
                source_id=str(numeric_id),
                num_instances=_safe_int(
                    _metadata_value(
                        metadata,
                        "num_instances",
                        None,
                    )
                ),
                num_features=_safe_int(
                    _metadata_value(
                        metadata,
                        "num_features",
                        None,
                    )
                ),

                # catalog_score는 인기도가 아니라
                # UCI 내부 검색 관련도이므로 retrieval_score에 저장한다.
                retrieval_score=catalog_score,
            )

        except Exception:
            return DatasetCandidate(
                dataset_name=_normalize_text(
                    item.get("name"),
                    default=f"UCI Dataset {numeric_id}",
                ),
                description="No description available.",
                task_type="Unknown",
                domain="Unknown",
                source=self.source_name,
                url=fallback_url,
                source_id=str(numeric_id),
                retrieval_score=catalog_score,
            )

    def search(
        self,
        project_description: str,
        limit: int = 10,
    ) -> list[DatasetCandidate]:
        """
        사용자 프로젝트 설명과 관련된 UCI 데이터셋을 반환한다.
        """

        query = project_description.strip()

        if not query or limit < 1:
            return []

        ranked_items = self._rank_catalog(
            query=query,
            limit=limit,
        )

        results: list[DatasetCandidate] = []

        for item in ranked_items:
            candidate = self._convert_item(item)

            if candidate is not None:
                results.append(candidate)

        return results


# 외부 함수 진입점에서 같은 인덱스를 재사용하기 위한 전역 인스턴스
_UCI_COLLECTOR = UCICollector()


def search_uci_datasets(
    project_description: str,
    limit: int = 10,
) -> list[DatasetCandidate]:
    """
    외부 코드에서 함수 형태로 UCI 검색을 호출하기 위한 진입점.
    """

    return _UCI_COLLECTOR.search(
        project_description=project_description,
        limit=limit,
    )