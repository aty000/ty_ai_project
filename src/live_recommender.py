from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.collectors.base import DatasetCollector
from src.collectors.common import DatasetCandidate
from src.collectors.kaggle_collector import KaggleCollector
from src.collectors.uci_collector import UCICollector
from src.dataset_analyzer import analyze_dataset


class SourceManager:
    """
    여러 데이터셋 저장소를 관리하고,
    각 저장소의 검색 결과를 하나의 후보 리스트로 통합한다.
    """

    def __init__(
        self,
        collectors: list[DatasetCollector] | None = None,
    ) -> None:
        if collectors is None:
            collectors = [
                UCICollector(),
                KaggleCollector(),
            ]

        self.collectors = collectors

    def search_all(
        self,
        project_description: str,
        limit_per_source: int = 5,
    ) -> list[DatasetCandidate]:
        """
        등록된 모든 Collector에서 데이터셋 후보를 검색한다.

        한 저장소가 실패해도 다른 저장소 검색은 계속한다.
        """

        query = project_description.strip()

        if not query or limit_per_source <= 0:
            return []

        all_candidates: list[DatasetCandidate] = []

        for collector in self.collectors:
            try:
                results = collector.search(
                    project_description=query,
                    limit=limit_per_source,
                )

                all_candidates.extend(results)

            except Exception as error:
                source_name = getattr(
                    collector,
                    "source_name",
                    collector.__class__.__name__,
                )

                print(
                    f"{source_name} collector failed: {error}"
                )

        return all_candidates


class LiveRecommender:
    """
    UCI와 Kaggle에서 후보를 수집하고,
    검색 점수와 통합 관련도를 분리하여 최종 순위를 계산한다.

    처리 단계:

    1. 후보 수집
    2. 동일 ID 중복 제거
    3. 전체 후보 관련도 계산
    4. 저장소별 retrieval 점수 정규화
    5. 최종 점수 계산
    6. 이름이 매우 유사한 후보 제거
    7. 상위 후보 분석
    """

    def __init__(
        self,
        source_manager: SourceManager | None = None,
        relevance_weight: float = 0.85,
        retrieval_weight: float = 0.15,
        minimum_relevance: float = 0.03,
        duplicate_name_threshold: float = 0.92,
    ) -> None:
        if relevance_weight < 0:
            raise ValueError(
                "relevance_weight는 0 이상이어야 합니다."
            )

        if retrieval_weight < 0:
            raise ValueError(
                "retrieval_weight는 0 이상이어야 합니다."
            )

        weight_sum = (
            relevance_weight
            + retrieval_weight
        )

        if weight_sum <= 0:
            raise ValueError(
                "점수 가중치의 합은 0보다 커야 합니다."
            )

        self.source_manager = (
            source_manager
            or SourceManager()
        )

        # 사용자가 다른 가중치를 넣더라도 합계가 1이 되도록 정규화
        self.relevance_weight = (
            relevance_weight / weight_sum
        )

        self.retrieval_weight = (
            retrieval_weight / weight_sum
        )

        self.minimum_relevance = self._clamp_score(
            minimum_relevance
        )

        self.duplicate_name_threshold = (
            self._clamp_score(
                duplicate_name_threshold
            )
        )

    def recommend(
        self,
        project_description: str,
        limit: int = 5,
        limit_per_source: int = 10,
    ) -> list[DatasetCandidate]:
        """
        데이터셋 후보를 수집하고 최종 추천 순위로 정렬한다.
        """

        query = project_description.strip()

        if (
            not query
            or limit <= 0
            or limit_per_source <= 0
        ):
            return []

        # 1. UCI와 Kaggle에서 후보 수집
        candidates = self.source_manager.search_all(
            project_description=query,
            limit_per_source=limit_per_source,
        )

        if not candidates:
            return []

        # 2. 동일 저장소의 동일 ID 후보 제거
        candidates = self._remove_exact_duplicates(
            candidates
        )

        if not candidates:
            return []

        # 3. 모든 후보를 같은 공간에서 비교하는 통합 관련도
        relevance_scores = self._calculate_relevance(
            query=query,
            candidates=candidates,
        )

        # 4. UCI와 Kaggle 내부 검색 점수 범위를 각각 0~1로 정규화
        retrieval_scores = (
            self._normalize_retrieval_scores_by_source(
                candidates
            )
        )

        ranked_candidates: list[
            tuple[
                float,
                float,
                float,
                DatasetCandidate,
            ]
        ] = []

        for (
            candidate,
            relevance,
            retrieval,
        ) in zip(
            candidates,
            relevance_scores,
            retrieval_scores,
            strict=True,
        ):
            relevance = self._clamp_score(
                relevance
            )

            retrieval = self._clamp_score(
                retrieval
            )

            if relevance < self.minimum_relevance:
                continue

            final_score = (
                relevance
                * self.relevance_weight
                + retrieval
                * self.retrieval_weight
            )

            candidate.relevance_score = round(
                relevance,
                6,
            )

            # Collector가 넣은 원점수 대신
            # 출처별로 정규화된 검색 점수를 출력한다.
            candidate.retrieval_score = round(
                retrieval,
                6,
            )

            candidate.final_score = round(
                final_score,
                6,
            )

            ranked_candidates.append(
                (
                    final_score,
                    relevance,
                    retrieval,
                    candidate,
                )
            )

        ranked_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
            reverse=True,
        )

        if not ranked_candidates:
            return []

        # 유사 이름 제거 후에도 limit개를 확보할 수 있도록
        # 최종 개수보다 넓게 후보를 가져온다.
        preselection_limit = min(
            len(ranked_candidates),
            max(
                limit * 4,
                limit,
            ),
        )

        preselected = [
            candidate
            for (
                _,
                _,
                _,
                candidate,
            ) in ranked_candidates[
                :preselection_limit
            ]
        ]

        # 5. 다른 저장소에 존재하는 동일·복제 데이터 억제
        unique_candidates = (
            self._remove_similar_name_duplicates(
                preselected
            )
        )

        results: list[DatasetCandidate] = []

        # 6. 최종 후보에만 상세 분석 적용
        for candidate in unique_candidates[:limit]:
            self._apply_analysis(candidate)
            results.append(candidate)

        return results

    @staticmethod
    def _calculate_relevance(
        query: str,
        candidates: list[DatasetCandidate],
    ) -> list[float]:
        """
        사용자 검색문과 데이터셋 메타데이터의
        TF-IDF 코사인 유사도를 계산한다.

        데이터셋 제목을 두 번 포함하여
        긴 설명보다 제목의 영향이 더 크도록 한다.
        """

        if not candidates:
            return []

        documents = [query]

        for candidate in candidates:
            dataset_text = " ".join(
                [
                    candidate.dataset_name,
                    candidate.dataset_name,
                    candidate.description,
                    candidate.task_type,
                    candidate.domain,
                ]
            )

            documents.append(dataset_text)

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )

            matrix = vectorizer.fit_transform(
                documents
            )

        except ValueError:
            return [
                0.0
                for _ in candidates
            ]

        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:],
        ).flatten()

        return [
            float(score)
            for score in scores
        ]

    def _normalize_retrieval_scores_by_source(
        self,
        candidates: list[DatasetCandidate],
    ) -> list[float]:
        """
        저장소마다 검색 점수의 범위가 다를 수 있으므로
        source별 최대값을 기준으로 0~1 범위로 정규화한다.

        기존 KaggleCollector가 검색 점수를 popularity에 저장한 경우를
        당장 깨뜨리지 않도록 호환 처리한다.
        """

        raw_scores: list[float] = []
        scores_by_source: dict[
            str,
            list[float],
        ] = defaultdict(list)

        for candidate in candidates:
            raw_score = self._get_raw_retrieval_score(
                candidate
            )

            raw_scores.append(raw_score)

            source_key = self._normalize_source(
                candidate.source
            )

            scores_by_source[source_key].append(
                raw_score
            )

        maximum_by_source: dict[str, float] = {}

        for source, scores in scores_by_source.items():
            maximum_by_source[source] = max(
                scores,
                default=0.0,
            )

        normalized_scores: list[float] = []

        for candidate, raw_score in zip(
            candidates,
            raw_scores,
            strict=True,
        ):
            source_key = self._normalize_source(
                candidate.source
            )

            source_maximum = maximum_by_source.get(
                source_key,
                0.0,
            )

            if source_maximum <= 0:
                normalized_scores.append(0.0)
                continue

            normalized_scores.append(
                self._clamp_score(
                    raw_score / source_maximum
                )
            )

        return normalized_scores

    @staticmethod
    def _get_raw_retrieval_score(
        candidate: DatasetCandidate,
    ) -> float:
        """
        Collector 내부 검색 점수를 반환한다.

        최신 Collector:
            retrieval_score 사용

        기존 KaggleCollector 호환:
            retrieval_score가 0이고 popularity가 존재하면
            popularity를 임시 검색 점수로 사용
        """

        retrieval = LiveRecommender._safe_float(
            candidate.retrieval_score
        )

        if retrieval > 0:
            return retrieval

        popularity = LiveRecommender._safe_float(
            candidate.popularity
        )

        if popularity > 0:
            return popularity

        return 0.0

    def _remove_exact_duplicates(
        self,
        candidates: list[DatasetCandidate],
    ) -> list[DatasetCandidate]:
        """
        동일한 source와 source_id를 가진 후보를 제거한다.

        같은 후보가 여러 번 들어온 경우
        저장소 내부 검색 점수가 더 높은 항목을 유지한다.
        """

        unique_by_id: dict[
            tuple[str, str],
            DatasetCandidate,
        ] = {}

        candidates_without_id: list[
            DatasetCandidate
        ] = []

        for candidate in candidates:
            source = self._normalize_source(
                candidate.source
            )

            source_id = str(
                candidate.source_id
            ).strip()

            if not source_id:
                candidates_without_id.append(
                    candidate
                )
                continue

            key = (
                source,
                source_id,
            )

            saved = unique_by_id.get(key)

            if saved is None:
                unique_by_id[key] = candidate
                continue

            candidate_score = (
                self._get_raw_retrieval_score(
                    candidate
                )
            )

            saved_score = (
                self._get_raw_retrieval_score(
                    saved
                )
            )

            if candidate_score > saved_score:
                unique_by_id[key] = candidate

        return [
            *unique_by_id.values(),
            *candidates_without_id,
        ]

    def _remove_similar_name_duplicates(
        self,
        candidates: list[DatasetCandidate],
    ) -> list[DatasetCandidate]:
        """
        최종 점수가 높은 후보부터 확인하여,
        이름이 거의 같은 데이터셋은 하나만 유지한다.

        candidates는 이미 final_score 내림차순으로 전달되므로
        먼저 저장된 후보가 더 높은 점수를 가진다.
        """

        unique: list[DatasetCandidate] = []

        for candidate in candidates:
            candidate_name = self._normalize_name(
                candidate.dataset_name
            )

            if not candidate_name:
                unique.append(candidate)
                continue

            duplicated = False

            for saved in unique:
                saved_name = self._normalize_name(
                    saved.dataset_name
                )

                if not saved_name:
                    continue

                if candidate_name == saved_name:
                    duplicated = True
                    break

                similarity = SequenceMatcher(
                    None,
                    candidate_name,
                    saved_name,
                ).ratio()

                if (
                    similarity
                    >= self.duplicate_name_threshold
                ):
                    duplicated = True
                    break

            if not duplicated:
                unique.append(candidate)

        return unique

    @staticmethod
    def _apply_analysis(
        candidate: DatasetCandidate,
    ) -> None:
        """
        데이터셋 이름과 설명을 기반으로 분석 결과를 적용한다.
        """

        analysis = analyze_dataset(
            dataset_name=candidate.dataset_name,
            description=candidate.description,
        )

        candidate.task_type = analysis.task_type
        candidate.domain = analysis.domain
        candidate.task_confidence = (
            analysis.confidence
        )

        candidate.recommended_metrics = (
            analysis.metrics
        )

        candidate.analysis_signals = (
            analysis.signals
        )

        candidate.target_variable = (
            analysis.target_variable
        )

        candidate.recommended_algorithms = (
            analysis.recommended_algorithms
        )

        candidate.difficulty = (
            analysis.difficulty
        )

        candidate.data_format = (
            analysis.data_format
        )

        candidate.warnings = analysis.warnings

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """
        데이터셋 이름 비교를 위해 소문자와 공백 형태로 정규화한다.
        """

        normalized = str(name).lower().strip()

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _normalize_source(
        source: str,
    ) -> str:
        return str(source).strip().lower()

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clamp_score(
        score: object,
    ) -> float:
        """
        값을 안전하게 0~1 범위로 제한한다.
        """

        numeric_score = LiveRecommender._safe_float(
            score
        )

        return min(
            max(numeric_score, 0.0),
            1.0,
        )