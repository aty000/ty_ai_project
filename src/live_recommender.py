from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity

from src.collectors.common import DatasetCandidate
from src.collectors.kaggle_collector import KaggleCollector
from src.collectors.uci_collector import UCICollector
from src.dataset_analyzer import (
    analyze_dataset,
    infer_query_intent,
    suggest_algorithms,
    suggest_metrics,
)


class LiveRecommender:
    """
    UCI와 Kaggle에서 데이터셋 후보를 수집하고,
    사용자 프로젝트 설명과의 적합도를 계산해 추천한다.

    처리 순서:
    1. 저장소별 후보 수집
    2. 정확한 중복 제거
    3. 후보 메타데이터 분석
    4. 텍스트 관련도 계산
    5. 저장소별 검색 점수 정규화
    6. Task 및 Domain 일치도 계산
    7. 최종 점수 계산
    8. 유사한 이름의 중복 제거
    """

    def __init__(self) -> None:
        self.collectors = [
            UCICollector(),
            KaggleCollector(),
        ]

    def recommend(
        self,
        project_description: str,
        limit: int = 5,
        limit_per_source: int = 10,
    ) -> list[DatasetCandidate]:
        """
        프로젝트 설명에 적합한 데이터셋을 추천한다.
        """

        query = project_description.strip()

        if not query:
            return []

        if limit < 1 or limit_per_source < 1:
            return []

        candidates = self._collect_candidates(
            project_description=query,
            limit_per_source=limit_per_source,
        )

        if not candidates:
            return []

        candidates = self._remove_exact_duplicates(
            candidates
        )

        self._apply_analysis(
            candidates
        )

        self._calculate_relevance_scores(
            project_description=query,
            candidates=candidates,
        )

        self._normalize_retrieval_scores(
            candidates
        )

        query_intent = infer_query_intent(
            query
        )

        self._calculate_match_scores(
            candidates=candidates,
            query_task=query_intent.task_type,
            query_domain=query_intent.domain,
        )

        self._calculate_final_scores(
            candidates
        )

        self._build_recommendation_reasons(
            candidates=candidates,
            query_task=query_intent.task_type,
            query_domain=query_intent.domain,
        )

        candidates.sort(
            key=lambda candidate: (
                candidate.final_score,
                candidate.relevance_score,
                candidate.task_match_score,
                candidate.domain_match_score,
                candidate.retrieval_score,
                candidate.popularity,
            ),
            reverse=True,
        )

        candidates = self._remove_similar_names(
            candidates
        )

        return candidates[:limit]

    def _collect_candidates(
        self,
        project_description: str,
        limit_per_source: int,
    ) -> list[DatasetCandidate]:
        """
        모든 Collector에서 데이터셋 후보를 수집한다.

        한 저장소에서 오류가 발생해도 다른 저장소 검색은 계속한다.
        """

        candidates: list[DatasetCandidate] = []

        for collector in self.collectors:
            try:
                source_candidates = collector.search(
                    project_description=project_description,
                    limit=limit_per_source,
                )

                candidates.extend(
                    source_candidates
                )

            except Exception as error:
                source_name = getattr(
                    collector,
                    "source_name",
                    collector.__class__.__name__,
                )

                print(
                    f"{source_name} collector failed: "
                    f"{error}"
                )

        return candidates

    @staticmethod
    def _remove_exact_duplicates(
        candidates: Iterable[DatasetCandidate],
    ) -> list[DatasetCandidate]:
        """
        이름이 완전히 동일한 후보를 하나로 합친다.

        동일한 이름이 여러 저장소에서 발견되면
        retrieval_score와 popularity가 높은 후보를 유지한다.
        """

        unique_candidates: dict[
            str,
            DatasetCandidate,
        ] = {}

        for candidate in candidates:
            normalized_name = (
                LiveRecommender._normalize_dataset_name(
                    candidate.dataset_name
                )
            )

            if not normalized_name:
                normalized_name = (
                    f"{candidate.source}:"
                    f"{candidate.source_id}:"
                    f"{candidate.url}"
                ).lower()

            existing = unique_candidates.get(
                normalized_name
            )

            if existing is None:
                unique_candidates[
                    normalized_name
                ] = candidate
                continue

            existing_strength = (
                existing.retrieval_score
                + existing.popularity
            )

            candidate_strength = (
                candidate.retrieval_score
                + candidate.popularity
            )

            if candidate_strength > existing_strength:
                unique_candidates[
                    normalized_name
                ] = candidate

        return list(
            unique_candidates.values()
        )

    @staticmethod
    def _apply_analysis(
        candidates: Iterable[DatasetCandidate],
    ) -> None:
        """
        모든 후보의 제목과 설명을 분석한다.

        Collector가 이미 제공한 Task와 Domain이 명확하면
        이를 유지하고, Unknown일 때만 분석 결과로 보완한다.
        """

        for candidate in candidates:
            analysis = analyze_dataset(
                dataset_name=candidate.dataset_name,
                description=candidate.description,
            )

            existing_task = (
                candidate.task_type.strip()
                if candidate.task_type
                else "Unknown"
            )

            existing_domain = (
                candidate.domain.strip()
                if candidate.domain
                else "Unknown"
            )

            if existing_task.lower() in {
                "",
                "unknown",
                "none",
                "not specified",
            }:
                candidate.task_type = (
                    analysis.task_type
                )

                candidate.task_confidence = (
                    analysis.confidence
                )
            else:
                candidate.task_type = (
                    existing_task
                )

                # 저장소가 제공한 Task는 분석 추론보다
                # 비교적 신뢰할 수 있다고 간주한다.
                candidate.task_confidence = max(
                    candidate.task_confidence,
                    0.80,
                )

            if existing_domain.lower() in {
                "",
                "unknown",
                "none",
                "not specified",
                "general",
            }:
                candidate.domain = (
                    analysis.domain
                )
            else:
                candidate.domain = (
                    existing_domain
                )

            candidate.recommended_metrics = (
                suggest_metrics(
                    candidate.task_type
                )
            )

            candidate.recommended_algorithms = (
                suggest_algorithms(
                    candidate.task_type
                )
            )

            candidate.analysis_signals = (
                analysis.signals
            )

            candidate.target_variable = (
                analysis.target_variable
            )

            candidate.difficulty = (
                analysis.difficulty
            )

            candidate.data_format = (
                analysis.data_format
            )

            candidate.warnings = (
                analysis.warnings
            )

    @staticmethod
    def _calculate_relevance_scores(
        project_description: str,
        candidates: list[DatasetCandidate],
    ) -> None:
        """
        사용자 프로젝트 설명과 각 데이터셋 메타데이터의
        TF-IDF 코사인 유사도를 계산한다.
        """

        if not candidates:
            return

        documents = [
            (
                f"{candidate.dataset_name} "
                f"{candidate.description} "
                f"{candidate.task_type} "
                f"{candidate.domain}"
            ).strip()
            for candidate in candidates
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
                [
                    project_description,
                    *documents,
                ]
            )

        except ValueError:
            for candidate in candidates:
                candidate.relevance_score = 0.0

            return

        query_vector = matrix[0:1]
        document_matrix = matrix[1:]

        scores = cosine_similarity(
            query_vector,
            document_matrix,
        ).flatten()

        for candidate, score in zip(
            candidates,
            scores,
        ):
            candidate.relevance_score = round(
                float(score),
                6,
            )

    @staticmethod
    def _normalize_retrieval_scores(
        candidates: Iterable[DatasetCandidate],
    ) -> None:
        """
        저장소마다 점수 분포가 다르기 때문에
        각 저장소 내부의 최대 점수를 기준으로 0~1 정규화한다.
        """

        grouped_candidates: dict[
            str,
            list[DatasetCandidate],
        ] = defaultdict(list)

        for candidate in candidates:
            grouped_candidates[
                candidate.source
            ].append(candidate)

        for source_candidates in (
            grouped_candidates.values()
        ):
            max_score = max(
                (
                    max(
                        candidate.retrieval_score,
                        0.0,
                    )
                    for candidate
                    in source_candidates
                ),
                default=0.0,
            )

            if max_score <= 0:
                for candidate in source_candidates:
                    candidate.retrieval_score = 0.0

                continue

            for candidate in source_candidates:
                normalized_score = (
                    max(
                        candidate.retrieval_score,
                        0.0,
                    )
                    / max_score
                )

                candidate.retrieval_score = round(
                    min(
                        normalized_score,
                        1.0,
                    ),
                    6,
                )

    @staticmethod
    def _calculate_match_scores(
        candidates: Iterable[DatasetCandidate],
        query_task: str,
        query_domain: str,
    ) -> None:
        """
        사용자 의도와 데이터셋의 Task 및 Domain 일치도를 계산한다.
        """

        normalized_query_task = (
            LiveRecommender._normalize_category(
                query_task
            )
        )

        normalized_query_domain = (
            LiveRecommender._normalize_category(
                query_domain
            )
        )

        for candidate in candidates:
            candidate_task = (
                LiveRecommender._normalize_category(
                    candidate.task_type
                )
            )

            candidate_domain = (
                LiveRecommender._normalize_category(
                    candidate.domain
                )
            )

            candidate.task_match_score = (
                LiveRecommender._category_match_score(
                    query_category=(
                        normalized_query_task
                    ),
                    candidate_category=(
                        candidate_task
                    ),
                    neutral_categories={
                        "unknown",
                    },
                )
            )

            candidate.domain_match_score = (
                LiveRecommender._category_match_score(
                    query_category=(
                        normalized_query_domain
                    ),
                    candidate_category=(
                        candidate_domain
                    ),
                    neutral_categories={
                        "unknown",
                        "general",
                    },
                )
            )

    @staticmethod
    def _category_match_score(
        query_category: str,
        candidate_category: str,
        neutral_categories: set[str],
    ) -> float:
        """
        Category 일치도를 0~1로 반환한다.

        사용자 의도를 추론하지 못한 경우에는
        후보를 과도하게 감점하지 않도록 중립 점수 0.5를 준다.
        """

        if query_category in neutral_categories:
            return 0.5

        if candidate_category in neutral_categories:
            return 0.0

        if query_category == candidate_category:
            return 1.0

        return 0.0

    @staticmethod
    def _calculate_final_scores(
        candidates: Iterable[DatasetCandidate],
    ) -> None:
        """
        각 평가 요소를 결합해 최종 추천 점수를 계산한다.

        가중치:
        - 텍스트 관련도: 65%
        - Task 일치도: 15%
        - Domain 일치도: 10%
        - 저장소 검색 점수: 10%

        명확한 Task 불일치에는 패널티를 적용한다.
        """

        for candidate in candidates:
            final_score = (
                candidate.relevance_score * 0.65
                + candidate.task_match_score * 0.15
                + candidate.domain_match_score * 0.10
                + candidate.retrieval_score * 0.10
            )

            # 사용자 Task와 데이터셋 Task가 명확히 다르면 감점
            if candidate.task_match_score == 0.0:
                final_score *= 0.35

            # Domain까지 불일치하면 추가 감점
            if candidate.domain_match_score == 0.0:
                final_score *= 0.70

            candidate.final_score = round(
                final_score,
                6,
            )

    @staticmethod
    def _build_recommendation_reasons(
        candidates: Iterable[DatasetCandidate],
        query_task: str,
        query_domain: str,
    ) -> None:
        """
        점수와 분석 결과를 바탕으로
        사용자가 이해할 수 있는 추천 이유를 생성한다.
        """

        normalized_query_task = (
            LiveRecommender._normalize_category(
                query_task
            )
        )

        normalized_query_domain = (
            LiveRecommender._normalize_category(
                query_domain
            )
        )

        for candidate in candidates:
            reasons: list[str] = []

            candidate_task = (
                LiveRecommender._normalize_category(
                    candidate.task_type
                )
            )

            candidate_domain = (
                LiveRecommender._normalize_category(
                    candidate.domain
                )
            )

            if (
                normalized_query_task
                not in {"", "unknown"}
                and candidate_task
                == normalized_query_task
            ):
                reasons.append(
                    f"Matches the requested "
                    f"{candidate.task_type} task."
                )

            if (
                normalized_query_domain
                not in {
                    "",
                    "unknown",
                    "general",
                }
                and candidate_domain
                == normalized_query_domain
            ):
                reasons.append(
                    f"Matches the requested "
                    f"{candidate.domain} domain."
                )

            if candidate.relevance_score >= 0.15:
                reasons.append(
                    "Dataset metadata is highly relevant "
                    "to the project description."
                )
            elif candidate.relevance_score >= 0.05:
                reasons.append(
                    "Dataset metadata contains terms "
                    "related to the project description."
                )

            if candidate.retrieval_score >= 0.75:
                reasons.append(
                    f"Ranked highly in the "
                    f"{candidate.source} search results."
                )
            elif candidate.retrieval_score >= 0.40:
                reasons.append(
                    f"Found as a relevant candidate "
                    f"in {candidate.source}."
                )

            if candidate.popularity >= 0.60:
                reasons.append(
                    "Has relatively strong popularity "
                    "or usage indicators."
                )
            elif candidate.popularity >= 0.40:
                reasons.append(
                    "Has moderate popularity "
                    "or usage indicators."
                )

            for signal in candidate.analysis_signals:
                reasons.append(signal)

            if candidate.data_format != "Unknown":
                reasons.append(
                    f"Expected data format: "
                    f"{candidate.data_format}."
                )

            if (
                candidate.relevance_score < 0.02
                and candidate.retrieval_score < 0.02
            ):
                reasons.append(
                    "Text relevance is weak, so the dataset "
                    "should be reviewed manually."
                )

            if not reasons:
                reasons.append(
                    "Included based on the combined "
                    "recommendation score."
                )

            candidate.recommendation_reasons = (
                reasons[:6]
            )

    @staticmethod
    def _remove_similar_names(
        candidates: Iterable[DatasetCandidate],
        similarity_threshold: float = 0.86,
    ) -> list[DatasetCandidate]:
        """
        이름이 매우 유사한 데이터셋을 중복 후보로 간주한다.

        candidates는 이미 최종 점수 순으로 정렬되어 있으므로
        먼저 등장한 높은 점수의 후보를 유지한다.
        """

        selected: list[DatasetCandidate] = []
        selected_names: list[str] = []

        for candidate in candidates:
            normalized_name = (
                LiveRecommender._normalize_dataset_name(
                    candidate.dataset_name
                )
            )

            if not normalized_name:
                selected.append(candidate)
                continue

            is_duplicate = any(
                SequenceMatcher(
                    None,
                    normalized_name,
                    existing_name,
                ).ratio()
                >= similarity_threshold
                for existing_name
                in selected_names
            )

            if is_duplicate:
                continue

            selected.append(candidate)
            selected_names.append(
                normalized_name
            )

        return selected

    @staticmethod
    def _normalize_dataset_name(
        dataset_name: str,
    ) -> str:
        """
        데이터셋 이름의 중복 비교를 위한 정규화.

        의미가 약한 일반 단어와 특수문자를 제거한다.
        """

        text = str(
            dataset_name
        ).lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        generic_words = {
            "data",
            "dataset",
            "datasets",
            "clean",
            "cleaned",
            "updated",
            "final",
            "version",
            "ml",
            "machine",
            "learning",
        }

        words = [
            word
            for word in text.split()
            if word not in generic_words
        ]

        return " ".join(words)

    @staticmethod
    def _normalize_category(
        value: str,
    ) -> str:
        """
        Task 또는 Domain 값을 비교 가능한 문자열로 정규화한다.
        """

        return re.sub(
            r"\s+",
            " ",
            str(value).strip().lower(),
        )