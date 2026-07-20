from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


UCI_LIST_API = "https://archive.ics.uci.edu/api/datasets/list"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "uci_catalog.csv"

DEFAULT_CACHE_DAYS = 7


def _is_cache_valid(
    path: Path,
    cache_days: int = DEFAULT_CACHE_DAYS,
) -> bool:
    if not path.exists():
        return False

    modified_at = datetime.fromtimestamp(
        path.stat().st_mtime
    )

    return datetime.now() - modified_at < timedelta(
        days=cache_days
    )


def download_uci_catalog() -> pd.DataFrame:
    """
    UCI의 Python 사용 가능 데이터셋 목록을 한 번 내려받는다.
    """

    try:
        response = requests.get(
            UCI_LIST_API,
            params={"filter": "python"},
            timeout=30,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

    except requests.RequestException as error:
        raise RuntimeError(
            f"UCI 카탈로그 다운로드 실패: {error}"
        ) from error

    except ValueError as error:
        raise RuntimeError(
            "UCI 카탈로그 응답을 JSON으로 해석하지 못했습니다."
        ) from error

    if payload.get("status") != 200:
        raise RuntimeError(
            f"UCI API 오류: {payload.get('message', 'Unknown error')}"
        )

    data = payload.get("data", [])

    if not isinstance(data, list):
        raise RuntimeError(
            "UCI 카탈로그 data 형식이 올바르지 않습니다."
        )

    dataframe = pd.DataFrame(data)

    required_columns = ["id", "name", "url"]

    for column in required_columns:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[required_columns].copy()

    dataframe["id"] = dataframe["id"].astype(str)
    dataframe["name"] = dataframe["name"].fillna("").astype(str)
    dataframe["url"] = dataframe["url"].fillna("").astype(str)

    return dataframe


def save_uci_catalog(
    dataframe: pd.DataFrame,
    path: Path = CATALOG_PATH,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def load_uci_catalog(
    force_refresh: bool = False,
    cache_days: int = DEFAULT_CACHE_DAYS,
) -> pd.DataFrame:
    """
    캐시가 있으면 로컬 CSV를 읽고,
    없거나 만료되었을 때만 UCI API를 호출한다.
    """

    if (
        not force_refresh
        and _is_cache_valid(
            CATALOG_PATH,
            cache_days=cache_days,
        )
    ):
        return pd.read_csv(
            CATALOG_PATH,
            dtype={"id": str},
        )

    try:
        dataframe = download_uci_catalog()
        save_uci_catalog(dataframe)
        return dataframe

    except RuntimeError:
        # API 장애 시 오래된 캐시라도 사용한다.
        if CATALOG_PATH.exists():
            return pd.read_csv(
                CATALOG_PATH,
                dtype={"id": str},
            )

        raise