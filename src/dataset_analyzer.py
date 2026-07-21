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


def _normalize_text(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> str:
    """
    데이터셋 제목, 설명, 태그를 하나의 소문자 문자열로 합친다.
    """
    return " ".join(
        [
            dataset_name,
            description,
            tags,
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

def infer_task_type(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> tuple[str, float]:
    """
    제목, 설명, 태그를 이용해 데이터셋의 작업 유형을 추론한다.
    """
    text = _normalize_text(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    task_keywords = {
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

    matched_scores: dict[str, int] = {}

    for task_type, keywords in task_keywords.items():
        score = sum(
            1
            for keyword in keywords
            if _contains_keyword(
                text=text,
                keyword=keyword,
            )
        )
        matched_scores[task_type] = score

    best_task = max(
        matched_scores,
        key=matched_scores.get,
    )
    best_score = matched_scores[best_task]

    if best_score == 0:
        return "Unknown", 0.0

    confidence = min(
        0.55 + best_score * 0.1,
        0.95,
    )

    return best_task, round(confidence, 2)

def infer_domain(
    dataset_name: str,
    description: str,
    tags: str = "",
) -> str:
    """
    제목, 설명, 태그를 이용해 데이터셋의 도메인을 추론한다.
    """
    text = _normalize_text(
        dataset_name=dataset_name,
        description=description,
        tags=tags,
    )

    domain_keywords = {
        "Real Estate": [
            "house",
            "housing",
            "property",
            "real estate",
            "rent",
            "mortgage",
        ],
        "Finance": [
            "finance",
            "financial",
            "stock",
            "bank",
            "credit",
            "loan",
            "fraud",
        ],
        "Healthcare": [
            "health",
            "medical",
            "patient",
            "disease",
            "diagnosis",
            "hospital",
        ],
        "Education": [
            "student",
            "education",
            "school",
            "university",
            "academic",
        ],
        "Energy": [
            "energy",
            "electricity",
            "power",
            "consumption",
            "solar",
            "wind",
        ],
        "Retail": [
            "retail",
            "sales",
            "customer",
            "purchase",
            "store",
            "shopping",
        ],
        "Transportation": [
            "traffic",
            "vehicle",
            "transport",
            "flight",
            "road",
            "mobility",
        ],
        "Text / NLP": [
            "text",
            "language",
            "sentiment",
            "review",
            "document",
            "nlp",
        ],
        "Image / Vision": [
            "image",
            "vision",
            "object detection",
            "segmentation",
            "photo",
        ],
    }

    domain_scores: dict[str, int] = {}

    for domain, keywords in domain_keywords.items():
        score = sum(
            1
            for keyword in keywords
            if _contains_keyword(
                text=text,
                keyword=keyword,
            )
        )
        domain_scores[domain] = score

    best_domain = max(
        domain_scores,
        key=domain_scores.get,
    )

    if domain_scores[best_domain] == 0:
        return "General"

    return best_domain


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
    작업 유형에 적합한 대표 머신러닝 알고리즘을 추천한다.
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
            "jpeg",
            "jpg",
            "png",
            "computer vision",
            "object detection",
        ],
        "Text": [
            "text",
            "document",
            "review",
            "sentence",
            "language",
            "nlp",
            "corpus",
        ],
        "Time Series": [
            "time series",
            "timestamp",
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

    format_scores: dict[str, int] = {}

    for data_format, keywords in format_keywords.items():
        score = sum(
            1
            for keyword in keywords
            if _contains_keyword(
                text=text,
                keyword=keyword,
            )
        )
        format_scores[data_format] = score

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
    분석에 도움이 될 만한 특징을 텍스트에서 추출한다.
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

    metrics = suggest_metrics(
        task_type=task_type,
    )

    algorithms = suggest_algorithms(
        task_type=task_type,
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

    target = "Unknown"
    difficulty = "Unknown"
    warnings = []

    return DatasetAnalysis(
        task_type=task_type,
        domain=domain,
        confidence=confidence,
        metrics=metrics,
        signals=signals,
        target_variable=target,
        recommended_algorithms=algorithms,
        difficulty=difficulty,
        data_format=data_format,
        warnings=warnings,
    )    