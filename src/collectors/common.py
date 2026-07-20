from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class DatasetCandidate:
    """
    서로 다른 데이터 저장소의 결과를 통합하는 공통 형식.
    """

    dataset_name: str
    description: str
    task_type: str
    domain: str
    source: str
    url: str

    source_id: str = ""
    num_instances: int | None = None
    num_features: int | None = None
    popularity: float = 0.0


def candidates_to_dataframe(
    candidates: list[DatasetCandidate],
) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "description",
        "task_type",
        "domain",
        "source",
        "url",
        "source_id",
        "num_instances",
        "num_features",
        "popularity",
    ]

    if not candidates:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        [asdict(candidate) for candidate in candidates],
        columns=columns,
    )