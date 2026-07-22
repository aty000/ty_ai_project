from __future__ import annotations

import math
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity

from src.collectors.base import DatasetCollector
from src.collectors.common import DatasetCandidate


class KaggleCollector(DatasetCollector):
    """
    Kaggle API에서 프로젝트 설명과 관련된 데이터셋을 검색한다.

    Kaggle API 검색 결과를 그대로 사용하지 않고,
    데이터셋 제목과 설명을 이용해 TF-IDF 관련도를 다시 계산한다.
    """

    source_name = "Kaggle"

    def __init__(self) -> None:
        self.api = KaggleApi()
        self._authenticated = False

    def _authenticate(self) -> None:
        """
        Kaggle API 인증을 최초 한 번만 실행한다.
        """

        if self._authenticated:
            return

        self.api.authenticate()
        self._authenticated = True

    def search(
        self,
        project_description: str,
        limit: int = 10,
    ) -> list[DatasetCandidate]:
        """
        Kaggle 데이터셋을 검색하고 로컬 관련도 점수를 계산한 뒤
        DatasetCandidate 목록으로 반환한다.
        """

        query = project_description.strip()

        if not query or limit < 1:
            return []

        # 최종 추천 개수보다 많은 후보를 가져온 뒤
        # 로컬 관련도 계산으로 다시 여과한다.
        candidate_pool_size = max(
            limit * 3,
            20,
        )

        try:
            self._authenticate()

            raw_datasets = self.api.dataset_list(
                search=query,
                sort_by="votes",
            )

            datasets = list(raw_datasets)[
                :candidate_pool_size
            ]

        except Exception as error:
            print(
                f"Kaggle search failed: {error}"
            )
            return []

        raw_candidates: list[
            dict[str, Any]
        ] = []

        for dataset in datasets:
            item = self._extract_dataset_item(
                dataset
            )

            # Kaggle 데이터셋 식별자가 없는 항목은
            # URL 및 중복 판정이 불가능하므로 제외한다.
            if not item["reference"]:
                continue

            raw_candidates.append(item)

        if not raw_candidates:
            return []

        retrieval_scores = (
            self._calculate_retrieval_scores(
                query=query,
                items=raw_candidates,
            )
        )

        candidates: list[
            DatasetCandidate
        ] = []

        for item, retrieval_score in zip(
            raw_candidates,
            retrieval_scores,
        ):
            candidates.append(
                DatasetCandidate(
                    dataset_name=item[
                        "dataset_name"
                    ],
                    description=item[
                        "description"
                    ],
                    task_type="Unknown",
                    domain="Unknown",
                    source=self.source_name,
                    url=item["url"],
                    source_id=item[
                        "reference"
                    ],
                    num_instances=None,
                    num_features=None,
                    popularity=item[
                        "popularity"
                    ],
                    retrieval_score=(
                        retrieval_score
                    ),
                )
            )

        # 검색 관련도를 우선하고,
        # 관련도가 같을 때 인기도를 보조 기준으로 사용한다.
        candidates.sort(
            key=lambda candidate: (
                candidate.retrieval_score,
                candidate.popularity,
            ),
            reverse=True,
        )

        return candidates[:limit]

    def _extract_dataset_item(
        self,
        dataset: object,
    ) -> dict[str, Any]:
        """
        Kaggle SDK 객체에서 필요한 메타데이터를 추출한다.
        """

        reference = self._normalize_text(
            self._get_value(
                dataset,
                "ref",
                "_ref",
                default="",
            ),
            default="",
        )

        dataset_name = self._normalize_text(
            self._get_value(
                dataset,
                "title",
                "_title",
                default="",
            ),
            default=(
                reference
                or "Unknown dataset"
            ),
        )

        description = self._normalize_text(
            self._get_value(
                dataset,
                "subtitle",
                "_subtitle",
                default="",
            ),
            default="",
        )

        if not description:
            description = self._normalize_text(
                self._get_value(
                    dataset,
                    "description",
                    "_description",
                    default="",
                ),
                default=(
                    "No description available."
                ),
            )

        vote_count = self._to_int(
            self._get_value(
                dataset,
                "vote_count",
                "voteCount",
                "_vote_count",
                default=0,
            )
        )

        download_count = self._to_int(
            self._get_value(
                dataset,
                "download_count",
                "downloadCount",
                "_download_count",
                default=0,
            )
        )

        view_count = self._to_int(
            self._get_value(
                dataset,
                "view_count",
                "viewCount",
                "_view_count",
                default=0,
            )
        )

        usability_rating = self._to_float(
            self._get_value(
                dataset,
                "usability_rating",
                "usabilityRating",
                "_usability_rating",
                default=0.0,
            )
        )

        popularity = (
            self._calculate_popularity(
                vote_count=vote_count,
                download_count=download_count,
                view_count=view_count,
                usability_rating=(
                    usability_rating
                ),
            )
        )

        raw_url = self._normalize_text(
            self._get_value(
                dataset,
                "url",
                "_url",
                default="",
            ),
            default="",
        )

        url = (
            raw_url
            or self._make_dataset_url(
                reference
            )
        )

        return {
            "reference": reference,
            "dataset_name": dataset_name,
            "description": description,
            "popularity": popularity,
            "url": url,
        }

    @staticmethod
    def _calculate_retrieval_scores(
        query: str,
        items: list[dict[str, Any]],
    ) -> list[float]:
        """
        사용자 질문과 Kaggle 데이터셋의 제목·설명 사이의
        TF-IDF 코사인 유사도를 계산한다.
        """

        search_documents = [
            (
                f"{item['dataset_name']} "
                f"{item['description']}"
            ).strip()
            for item in items
        ]

        custom_stop_words = (
            ENGLISH_STOP_WORDS.union(
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
        )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(
                custom_stop_words
            ),
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        try:
            matrix = vectorizer.fit_transform(
                [query, *search_documents]
            )

        except ValueError:
            # 모든 문서가 불용어뿐이거나
            # 유효한 단어가 없는 경우
            return [
                0.0
                for _ in items
            ]

        query_vector = matrix[0:1]
        document_matrix = matrix[1:]

        scores = cosine_similarity(
            query_vector,
            document_matrix,
        ).flatten()

        return [
            round(
                float(score),
                6,
            )
            for score in scores
        ]

    @staticmethod
    def _get_value(
        obj: object,
        *attribute_names: str,
        default: object = None,
    ) -> object:
        """
        여러 후보 속성명 중 실제로 존재하는 첫 번째 값을 반환한다.

        Kaggle SDK 버전에 따라 다음 속성 형태가 섞일 수 있다.

        - vote_count
        - voteCount
        - _vote_count
        """

        for attribute_name in (
            attribute_names
        ):
            try:
                value = getattr(
                    obj,
                    attribute_name,
                )
            except (
                AttributeError,
                TypeError,
            ):
                continue

            if value is not None:
                return value

        return default

    @staticmethod
    def _normalize_text(
        value: object,
        default: str = "",
    ) -> str:
        """
        Kaggle SDK에서 가져온 값을 안전한 문자열로 변환한다.
        """

        if value is None:
            return default

        text = str(value).strip()

        return text if text else default

    @staticmethod
    def _make_dataset_url(
        reference: str,
    ) -> str:
        """
        Kaggle 데이터셋 식별자를 URL로 변환한다.
        """

        if not reference:
            return (
                "https://www.kaggle.com/datasets"
            )

        return (
            "https://www.kaggle.com/datasets/"
            f"{reference}"
        )

    @staticmethod
    def _to_int(
        value: object,
    ) -> int:
        """
        Kaggle 메타데이터 값을 정수로 변환한다.
        """

        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def _to_float(
        value: object,
    ) -> float:
        """
        Kaggle 메타데이터 값을 실수로 변환한다.
        """

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _calculate_popularity(
        vote_count: int,
        download_count: int,
        view_count: int,
        usability_rating: float,
    ) -> float:
        """
        Kaggle의 인기도와 품질 지표를 0~1 점수로 통합한다.

        가중치:
        - 투표 수: 35%
        - 다운로드 수: 30%
        - 조회 수: 20%
        - 사용성 점수: 15%
        """

        vote_score = min(
            math.log1p(
                max(vote_count, 0)
            )
            / math.log1p(10_000),
            1.0,
        )

        download_score = min(
            math.log1p(
                max(download_count, 0)
            )
            / math.log1p(1_000_000),
            1.0,
        )

        view_score = min(
            math.log1p(
                max(view_count, 0)
            )
            / math.log1p(10_000_000),
            1.0,
        )

        usability_score = min(
            max(
                usability_rating,
                0.0,
            ),
            1.0,
        )

        score = (
            vote_score * 0.35
            + download_score * 0.30
            + view_score * 0.20
            + usability_score * 0.15
        )

        return round(
            score,
            6,
        )


_KAGGLE_COLLECTOR = KaggleCollector()


def search_kaggle_datasets(
    project_description: str,
    limit: int = 10,
) -> list[DatasetCandidate]:
    """
    테스트와 외부 모듈에서 사용하는 간편 호출 함수.
    """

    return _KAGGLE_COLLECTOR.search(
        project_description=(
            project_description
        ),
        limit=limit,
    )