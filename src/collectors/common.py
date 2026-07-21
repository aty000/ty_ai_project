from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetCandidate:
    dataset_name: str
    description: str
    task_type: str
    domain: str
    source: str
    url: str

    source_id: str = ""

    num_instances: int | None = None
    num_features: int | None = None

    # 실제 인기도나 다운로드 수 등을 0~1로 변환한 값
    popularity: float = 0.0

    # 각 저장소 내부에서 후보를 검색할 때 계산된 점수
    retrieval_score: float = 0.0

    # 모든 저장소 후보를 합친 뒤 사용자 질문과 다시 계산한 관련도
    relevance_score: float = 0.0

    # retrieval_score와 relevance_score 등을 결합한 최종 추천 점수
    final_score: float = 0.0

    task_confidence: float = 0.0

    recommended_metrics: list[str] = field(
        default_factory=list
    )

    analysis_signals: list[str] = field(
        default_factory=list
    )

    target_variable: str = "Unknown"

    recommended_algorithms: list[str] = field(
        default_factory=list
    )

    difficulty: str = "Unknown"

    data_format: str = "Unknown"

    warnings: list[str] = field(
        default_factory=list
    )