from __future__ import annotations

import math

from kaggle.api.kaggle_api_extended import KaggleApi

from src.collectors.base import DatasetCollector
from src.collectors.common import DatasetCandidate


class KaggleCollector(DatasetCollector):
    """
    Kaggle API에서 프로젝트 설명과 관련된 데이터셋을 검색한다.
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
        Kaggle 데이터셋을 검색한 뒤 DatasetCandidate 목록으로 변환한다.
        """
        query = project_description.strip()

        if not query or limit <= 0:
            return []

        try:
            self._authenticate()

            datasets = self.api.dataset_list(
                search=query,
                sort_by="votes",
            )

        except Exception as error:
            print(f"Kaggle search failed: {error}")
            return []

        candidates: list[DatasetCandidate] = []

        for dataset in datasets[:limit]:
            reference = str(
                self._get_value(
                    dataset,
                    "ref",
                    "_ref",
                    default="",
                )
                or ""
            )

            dataset_name = str(
                self._get_value(
                    dataset,
                    "title",
                    "_title",
                    default="",
                )
                or reference
                or "Unknown dataset"
            )

            description = str(
                self._get_value(
                    dataset,
                    "subtitle",
                    "_subtitle",
                    default="",
                )
                or self._get_value(
                    dataset,
                    "description",
                    "_description",
                    default="",
                )
                or ""
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

            popularity = self._calculate_popularity(
                vote_count=vote_count,
                download_count=download_count,
                view_count=view_count,
                usability_rating=usability_rating,
            )

            url = str(
                self._get_value(
                    dataset,
                    "url",
                    "_url",
                    default="",
                )
                or self._make_dataset_url(reference)
            )

            candidates.append(
                DatasetCandidate(
                    dataset_name=dataset_name,
                    description=description,
                    task_type="Unknown",
                    domain="Unknown",
                    source=self.source_name,
                    url=url,
                    source_id=reference,
                    num_instances=None,
                    num_features=None,
                    popularity=popularity,
                )
            )

        return candidates

    @staticmethod
    def _get_value(
        obj: object,
        *attribute_names: str,
        default: object = None,
    ) -> object:
        """
        여러 후보 속성명 중 실제로 존재하는 첫 번째 값을 반환한다.

        Kaggle SDK 버전에 따라 다음 형태가 섞여 있을 수 있다.
        - vote_count
        - voteCount
        - _vote_count
        """
        for attribute_name in attribute_names:
            try:
                value = getattr(obj, attribute_name)
            except (AttributeError, TypeError):
                continue

            if value is not None:
                return value

        return default

    @staticmethod
    def _make_dataset_url(reference: str) -> str:
        if not reference:
            return "https://www.kaggle.com/datasets"

        return f"https://www.kaggle.com/datasets/{reference}"

    @staticmethod
    def _to_int(value: object) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _to_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
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
            math.log1p(max(vote_count, 0))
            / math.log1p(10_000),
            1.0,
        )

        download_score = min(
            math.log1p(max(download_count, 0))
            / math.log1p(1_000_000),
            1.0,
        )

        view_score = min(
            math.log1p(max(view_count, 0))
            / math.log1p(10_000_000),
            1.0,
        )

        usability_score = min(
            max(usability_rating, 0.0),
            1.0,
        )

        score = (
            vote_score * 0.35
            + download_score * 0.30
            + view_score * 0.20
            + usability_score * 0.15
        )

        return round(score, 6)


_kaggle_collector = KaggleCollector()


def search_kaggle_datasets(
    project_description: str,
    limit: int = 10,
) -> list[DatasetCandidate]:
    """
    테스트와 외부 모듈에서 사용하는 간편 호출 함수.
    """
    return _kaggle_collector.search(
        project_description=project_description,
        limit=limit,
    )