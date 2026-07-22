from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetCandidate:
    # 기본 데이터셋 정보
    dataset_name: str
    description: str
    task_type: str
    domain: str
    source: str
    url: str

    # 저장소 내부 식별자
    source_id: str = ""

    # 데이터 규모
    num_instances: int | None = None
    num_features: int | None = None

    # 실제 인기도나 다운로드 수 등을 0~1로 변환한 값
    popularity: float = 0.0

    # 각 저장소 내부에서 후보를 검색할 때 계산된 점수
    retrieval_score: float = 0.0

    # 모든 저장소 후보를 합친 뒤
    # 사용자 질문과 다시 계산한 텍스트 관련도
    relevance_score: float = 0.0

    # 사용자 질문에서 추론한 Task와
    # 데이터셋 Task가 일치하는 정도
    task_match_score: float = 0.0

    # 사용자 질문에서 추론한 Domain과
    # 데이터셋 Domain이 일치하는 정도
    domain_match_score: float = 0.0

    # 여러 점수를 결합한 최종 추천 점수
    final_score: float = 0.0

    # 데이터셋 Task 추론 신뢰도
    task_confidence: float = 0.0

    # 분석 결과
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

    recommendation_reasons: list[str] = field(
        default_factory=list
    )