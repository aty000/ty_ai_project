from __future__ import annotations

import re


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "dataset",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "machine",
    "learning",
    "model",
    "of",
    "on",
    "project",
    "the",
    "to",
    "using",
    "want",
    "with",
}


def build_search_query(
    project_description: str,
    max_keywords: int = 6,
) -> str:
    """
    긴 프로젝트 설명에서 외부 검색에 사용할
    핵심 영문 키워드를 추출한다.

    예:
    I want to predict house prices using income and location data.

    결과:
    predict house prices income location data
    """

    words = re.findall(
        r"[A-Za-z0-9]+",
        project_description.lower(),
    )

    keywords: list[str] = []

    for word in words:
        if len(word) < 3:
            continue

        if word in STOP_WORDS:
            continue

        if word in keywords:
            continue

        keywords.append(word)

        if len(keywords) >= max_keywords:
            break

    return " ".join(keywords)