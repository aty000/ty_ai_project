from __future__ import annotations

from abc import ABC, abstractmethod

from src.collectors.common import DatasetCandidate


class DatasetCollector(ABC):
    """
    모든 데이터 저장소 수집기가 따라야 하는 공통 규격.

    각 저장소는 검색 방식이 달라도,
    최종적으로 DatasetCandidate 목록을 반환해야 한다.
    """

    source_name: str

    @abstractmethod
    def search(
        self,
        project_description: str,
        limit: int = 10,
    ) -> list[DatasetCandidate]:
        """
        저장소에서 데이터셋 후보를 검색한다.
        """
        raise NotImplementedError