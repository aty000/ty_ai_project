from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DatasetAnalysis:
    task_type: str
    domain: str
    confidence: float

    metrics: list[str]
    signals: list[str]

    target_variable: str
    recommended_algorithms: list[str]

    difficulty: str
    data_format: str

    warnings: list[str]


@dataclass
class QueryIntent:
    """
    사용자의 프로젝트 설명에서 추론한 검색 의도.
    """

    task_type: str
    domain: str
    task_confidence: float


TASK_KEYWORDS: dict[str, list[str]] = {
    "Time Series": [
        "time series",
        "forecast",
        "forecasting",
        "hourly",
        "daily",
        "weekly",
        "monthly",
        "yearly",
        "timestamp",
        "timestamps",
        "temporal",
        "trend",
        "trends",
    ],
    "Classification": [
        "classification",
        "classify",
        "category",
        "categories",
        "categorical target",
        "binary classification",
        "binary target",
        "multiclass",
        "class label",
        "class labels",
        "target class",
        "predict whether",
        "whether income",
        "income exceeds",
        "income class",
        "income level",
        "above 50k",
        "greater than 50k",
        ">50k",
        "<=50k",
        "fraud detection",
        "diagnosis",
        "churn",
        "spam",
        "sentiment",
        "survival",
        "default prediction",
        "loan approval",
        "label",
        "labels",
    ],
    "Regression": [
        "regression",
        "predict price",
        "predict prices",
        "predicting price",
        "predicting prices",
        "price prediction",
        "price predictions",
        "house price",
        "house prices",
        "housing price",
        "housing prices",
        "property price",
        "property prices",
        "home price",
        "home prices",

        "housing market",
        "housing market factors",
        "real estate valuation",
        "property valuation",
        "home value",
        "home values",
        "median house value",
        "median housing value",

        "sales prediction",
        "sales predictions",
        "income prediction",
        "income predictions",
        "demand prediction",
        "demand predictions",
        "continuous target",
        "continuous value",
        "continuous values",
        "numeric target",
        "numerical target",
    ],
    "Clustering": [
        "clustering",
        "cluster",
        "clusters",
        "segmentation",
        "customer segmentation",
        "grouping",
        "unsupervised",
    ],
    "Computer Vision": [
        "image",
        "images",
        "computer vision",
        "object detection",
        "image classification",
        "segmentation mask",
        "segmentation masks",
        "bounding box",
        "bounding boxes",
    ],
    "NLP": [
        "text",
        "texts",
        "natural language",
        "nlp",
        "sentiment analysis",
        "language model",
        "language models",
        "document classification",
        "review text",
        "review texts",
    ],
}


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Real Estate": [
        "house",
        "houses",
        "housing",
        "home price",
        "home prices",
        "property",
        "properties",
        "real estate",
        "rent",
        "rental",
        "mortgage",
        "apartment",
        "apartments",
    ],
    "Finance": [
        "finance",
        "financial",
        "stock",
        "stocks",
        "bank",
        "banking",
        "credit",
        "loan",
        "loans",
        "fraud",
        "insurance",
    ],
    "Healthcare": [
        "health",
        "healthcare",
        "medical",
        "patient",
        "patients",
        "disease",
        "diagnosis",
        "hospital",
        "clinical",
    ],
    "Education": [
        "student",
        "students",
        "education",
        "school",
        "university",
        "academic",
        "learning outcome",
    ],
    "Energy": [
        "energy",
        "electricity",
        "power",
        "consumption",
        "solar",
        "wind",
        "grid",
    ],
    "Retail": [
        "retail",
        "sales",
        "customer",
        "customers",
        "purchase",
        "purchases",
        "store",
        "shopping",
        "ecommerce",
        "e-commerce",
    ],
    "Transportation": [
        "traffic",
        "vehicle",
        "vehicles",
        "transport",
        "transportation",
        "flight",
        "flights",
        "road",
        "mobility",
    ],
    "Text / NLP": [
        "text",
        "language",
        "sentiment",
        "review",
        "reviews",
        "document",
        "documents",
        "nlp",
    ],
    "Image / Vision": [
        "image",
        "images",
        "vision",
        "object detection",
        "segmentation",
        "photo",
        "photos",
    ],
}


def _normalize_text(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> str:
    """
    제목, 설명, 태그를 하나의 정규화된 문자열로 합친다.
    """

    return " ".join(
        [
            str(dataset_name),
            str(description),
            str(tags),
        ]
    ).lower()


def _contains_keyword(
    text: str,
    keyword: str,
) -> bool:
    """
    키워드가 다른 단어의 일부가 아닌
    독립된 단어 또는 구문으로 존재하는지 확인한다.
    """

    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def _score_keyword_groups(
    text: str,
    keyword_groups: dict[str, list[str]],
) -> dict[str, int]:
    """
    각 Task 또는 Domain에 해당하는 키워드 일치 개수를 계산한다.
    """

    scores: dict[str, int] = {}

    for category, keywords in keyword_groups.items():
        scores[category] = sum(
            1
            for keyword in keywords
            if _contains_keyword(
                text=text,
                keyword=keyword,
            )
        )

    return scores


def infer_task_type(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> tuple[str, float]:
    """
    제목, 설명, 태그를 이용해 작업 유형을 추론한다.
    """

    text = _normalize_text(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    matched_scores = _score_keyword_groups(
        text=text,
        keyword_groups=TASK_KEYWORDS,
    )

    best_task = max(
        matched_scores,
        key=matched_scores.get,
    )

    best_score = matched_scores[best_task]

    if best_score == 0:
        return "Unknown", 0.0

    sorted_scores = sorted(
        matched_scores.values(),
        reverse=True,
    )

    second_score = (
        sorted_scores[1]
        if len(sorted_scores) > 1
        else 0
    )

    # 가장 높은 Task와 두 번째 Task의 차이가 클수록
    # 추론 신뢰도를 높인다.
    score_gap = max(
        best_score - second_score,
        0,
    )

    confidence = min(
        0.50
        + best_score * 0.10
        + score_gap * 0.05,
        0.95,
    )

    return (
        best_task,
        round(confidence, 2),
    )


def infer_domain(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> str:
    """
    제목, 설명, 태그를 이용해 데이터 도메인을 추론한다.
    """

    text = _normalize_text(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    domain_scores = _score_keyword_groups(
        text=text,
        keyword_groups=DOMAIN_KEYWORDS,
    )

    best_domain = max(
        domain_scores,
        key=domain_scores.get,
    )

    if domain_scores[best_domain] == 0:
        return "General"

    return best_domain


def infer_query_intent(
    project_description: str,
) -> QueryIntent:
    """
    사용자 프로젝트 설명에서 원하는 Task와 Domain을 추론한다.
    """

    query = project_description.strip()

    if not query:
        return QueryIntent(
            task_type="Unknown",
            domain="General",
            task_confidence=0.0,
        )

    task_type, task_confidence = infer_task_type(
        dataset_name=query,
        description="",
    )

    domain = infer_domain(
        dataset_name=query,
        description="",
    )

    return QueryIntent(
        task_type=task_type,
        domain=domain,
        task_confidence=task_confidence,
    )


def suggest_metrics(
    task_type: str,
) -> list[str]:
    """
    작업 유형에 맞는 대표 평가 지표를 반환한다.
    """

    metric_map = {
        "Regression": [
            "MAE",
            "RMSE",
            "R²",
        ],
        "Classification": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ],
        "Time Series": [
            "MAE",
            "RMSE",
            "MAPE",
        ],
        "Clustering": [
            "Silhouette Score",
            "Davies-Bouldin Index",
        ],
        "Computer Vision": [
            "Accuracy",
            "Precision",
            "Recall",
            "mAP",
        ],
        "NLP": [
            "Accuracy",
            "F1-score",
            "BLEU / ROUGE",
        ],
    }

    return metric_map.get(
        task_type,
        ["Metric unavailable"],
    )


def suggest_algorithms(
    task_type: str,
) -> list[str]:
    """
    작업 유형에 적합한 대표 머신러닝 알고리즘을 반환한다.
    """

    algorithm_map = {
        "Regression": [
            "Linear Regression",
            "Random Forest Regressor",
            "XGBoost Regressor",
            "LightGBM Regressor",
        ],
        "Classification": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
        ],
        "Time Series": [
            "ARIMA",
            "Prophet",
            "LSTM",
            "XGBoost",
        ],
        "Clustering": [
            "K-Means",
            "DBSCAN",
            "Hierarchical Clustering",
        ],
        "Computer Vision": [
            "ResNet",
            "EfficientNet",
            "YOLO",
        ],
        "NLP": [
            "BERT",
            "RoBERTa",
            "DistilBERT",
        ],
    }

    return algorithm_map.get(
        task_type,
        ["Algorithm unavailable"],
    )


def infer_data_format(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> str:
    """
    데이터셋 메타데이터에서 예상 데이터 형식을 추론한다.
    """

    text = _normalize_text(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    format_keywords = {
        "Image": [
            "image",
            "images",
            "photo",
            "photos",
            "jpeg",
            "jpg",
            "png",
            "computer vision",
            "object detection",
        ],
        "Text": [
            "text",
            "document",
            "documents",
            "review",
            "reviews",
            "sentence",
            "language",
            "nlp",
            "corpus",
        ],
        "Time Series": [
            "time series",
            "timestamp",
            "timestamps",
            "temporal",
            "hourly",
            "daily",
            "weekly",
            "monthly",
            "yearly",
            "forecast",
        ],
        "Tabular": [
            "csv",
            "table",
            "tables",
            "tabular",
            "structured",
            "row",
            "rows",
            "column",
            "columns",
            "record",
            "records",
            "instance",
            "instances",
            "sample",
            "samples",
            "attribute",
            "attributes",
            "feature",
            "features",
            "observation",
            "observations",
            "numeric variables",
            "categorical variables",
        ],
    }

    format_scores = _score_keyword_groups(
        text=text,
        keyword_groups=format_keywords,
    )

    best_format = max(
        format_scores,
        key=format_scores.get,
    )

    if format_scores[best_format] == 0:
        return "Unknown"

    return best_format


def extract_analysis_signals(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> list[str]:
    """
    분석에 도움이 되는 데이터 특징을 텍스트에서 추출한다.
    """

    text = _normalize_text(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    signal_keywords = {
        "Price target likely available": [
            "price",
            "prices",
            "cost",
            "costs",
            "value",
            "values",
            "house price",
            "house prices",
            "housing price",
            "housing prices",
            "property price",
            "property prices",
        ],
        "Income feature detected": [
            "income",
            "incomes",
            "salary",
            "salaries",
            "earning",
            "earnings",
            "wage",
            "wages",
        ],
        "Location feature detected": [
            "location",
            "locations",
            "latitude",
            "latitudes",
            "longitude",
            "longitudes",
            "region",
            "regions",
            "city",
            "cities",
            "state",
            "states",
            "address",
            "addresses",
        ],
        "Time-related feature detected": [
            "date",
            "dates",
            "time",
            "times",
            "timestamp",
            "timestamps",
            "hourly",
            "daily",
            "weekly",
            "monthly",
            "yearly",
            "temporal",
        ],
        "Demographic features detected": [
            "age",
            "ages",
            "gender",
            "genders",
            "population",
            "populations",
            "demographic",
            "demographics",
            "occupation",
            "occupations",
            "education level",
            "education levels",
        ],
        "Likely tabular data": [
            "structured",
            "table",
            "tables",
            "csv",
            "tabular",
            "row",
            "rows",
            "column",
            "columns",
            "record",
            "records",
            "feature",
            "features",
            "attribute",
            "attributes",
            "instance",
            "instances",
        ],
    }

    signals: list[str] = []

    for signal, keywords in signal_keywords.items():
        if any(
            _contains_keyword(
                text=text,
                keyword=keyword,
            )
            for keyword in keywords
        ):
            signals.append(signal)

    return signals


def infer_target_variable(
    dataset_name: str,
    description: str,
) -> str:
    """
    현재 단계에서 명확하게 식별 가능한 대표 타깃만 추론한다.
    """

    text = _normalize_text(
        dataset_name=dataset_name,
        description=description,
    )

    target_keywords = {
        "Price": [
            "house price",
            "house prices",
            "housing price",
            "housing prices",
            "property price",
            "property prices",
            "sale price",
            "saleprice",
        ],
        "Income": [
            "income prediction",
            "predict income",
            "income target",
        ],
        "Sales": [
            "sales prediction",
            "predict sales",
            "sales target",
        ],
        "Demand": [
            "demand prediction",
            "predict demand",
            "demand target",
        ],
    }

    scores = _score_keyword_groups(
        text=text,
        keyword_groups=target_keywords,
    )

    best_target = max(
        scores,
        key=scores.get,
    )

    if scores[best_target] == 0:
        return "Unknown"

    return best_target


def infer_difficulty(
    task_type: str,
    data_format: str,
) -> str:
    """
    Task와 데이터 형식을 이용한 단순 난이도 추정.
    """

    if task_type in {
        "Computer Vision",
        "NLP",
        "Time Series",
    }:
        return "Intermediate"

    if data_format == "Tabular":
        return "Beginner"

    return "Unknown"


def build_warnings(
    task_type: str,
    domain: str,
    data_format: str,
) -> list[str]:
    """
    메타데이터 분석 결과가 불확실할 때 경고를 생성한다.
    """

    warnings: list[str] = []

    if task_type == "Unknown":
        warnings.append(
            "Task type could not be inferred from metadata."
        )

    if domain == "General":
        warnings.append(
            "Domain could not be identified precisely."
        )

    if data_format == "Unknown":
        warnings.append(
            "Data format should be checked before use."
        )

    return warnings


def analyze_dataset(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> DatasetAnalysis:
    """
    데이터셋 메타데이터를 종합 분석한다.
    """

    task_type, confidence = infer_task_type(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    domain = infer_domain(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    data_format = infer_data_format(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    signals = extract_analysis_signals(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    target = infer_target_variable(
        dataset_name=dataset_name,
        description=description,
    )

    difficulty = infer_difficulty(
        task_type=task_type,
        data_format=data_format,
    )

    warnings = build_warnings(
        task_type=task_type,
        domain=domain,
        data_format=data_format,
    )

    return DatasetAnalysis(
        task_type=task_type,
        domain=domain,
        confidence=confidence,
        metrics=suggest_metrics(task_type),
        signals=signals,
        target_variable=target,
        recommended_algorithms=(
            suggest_algorithms(task_type)
        ),
        difficulty=difficulty,
        data_format=data_format,
        warnings=warnings,
    )