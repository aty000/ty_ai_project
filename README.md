# AI Dataset Recommendation System

> A web-based recommendation system that helps users discover the most suitable datasets for AI projects by integrating UCI Machine Learning Repository and Kaggle.

AI Dataset Recommendation System은 AI 프로젝트에 적합한 데이터셋을 효율적으로 탐색하고 추천하기 위한 웹 기반 서비스입니다.

사용자는 프로젝트에 대한 간단한 설명만 입력하면 UCI Machine Learning Repository와 Kaggle에서 관련 데이터셋을 검색하고, **TF-IDF 기반 유사도 분석**을 통해 프로젝트에 적합한 데이터셋을 추천받을 수 있습니다.

또한 추천 결과와 함께 **추천 이유(Recommendation Reason)**, **적합한 머신러닝 알고리즘(Recommended ML Algorithm)**, **평가 지표(Evaluation Metric)**를 제공하여 데이터셋 선정부터 모델 개발까지의 의사결정을 지원합니다.

---

# 1. Project Overview

## 프로젝트 소개

AI 프로젝트를 시작할 때 가장 먼저 필요한 것은 프로젝트 목적에 적합한 데이터셋입니다.

본 프로젝트는 여러 데이터셋 플랫폼을 통합 검색하고 프로젝트 설명을 기반으로 관련성이 높은 데이터셋을 추천하여 데이터 탐색 시간을 줄이고 효율적인 프로젝트 시작을 지원합니다.

---

# 2. Motivation

## 프로젝트 개발 배경

AI 프로젝트를 수행할 때 적합한 데이터셋을 찾기 위해서는 여러 데이터 플랫폼을 직접 검색하고 데이터셋의 특성과 품질을 비교해야 합니다.

하지만 이러한 과정은 많은 시간이 소요되며, 특히 초보자는 어떤 데이터셋이 자신의 프로젝트에 적합한지 판단하기 어렵습니다.

본 프로젝트는 이러한 문제를 해결하기 위해 **프로젝트 설명 기반 데이터셋 추천 시스템**을 구현하였습니다.

---

# 3. Recommendation Process

## Overall Workflow

```text
User Input
      │
Project Analysis
      │
 ┌────┴────┐
 │         │
UCI     Kaggle
 │         │
 └────┬────┘
      │
Candidate Datasets
      │
Dataset Metadata Collection
      │
TF-IDF Similarity
      │
Recommendation Score
      │
Recommendation Results
 ├── Recommendation Reason
 ├── Recommended ML Algorithm
 ├── Evaluation Metric
 └── Dataset Comparison
```

### Step 1. User Input

사용자는 AI 프로젝트에 대한 간단한 설명을 입력합니다.

**Example**

- House Price Prediction
- Medical Image Classification
- Customer Churn Prediction

### Step 2. Dataset Search

프로젝트 설명을 기반으로 다음 플랫폼에서 관련 데이터셋을 검색합니다.

- UCI Machine Learning Repository
- Kaggle

### Step 3. Dataset Metadata Collection

검색된 데이터셋의 메타데이터를 수집합니다.

- Dataset Name
- Description
- Task Type
- Domain
- Number of Instances
- Number of Features
- Popularity
- Source

### Step 4. Similarity Analysis

사용자 입력과 데이터셋 설명을 TF-IDF 기반으로 벡터화하고 Cosine Similarity를 이용하여 관련성을 계산합니다.

### Step 5. Recommendation

유사도와 메타데이터를 기반으로 추천 점수를 계산한 후 가장 적합한 데이터셋을 추천합니다.

---

# 4. Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Web Framework | Streamlit |
| Machine Learning | scikit-learn |
| Data Processing | Pandas |
| Dataset Source | UCI Machine Learning Repository |
| Dataset Source | Kaggle API |
| Version Control | Git / GitHub |

---

# 5. Project Structure

```text
AI-Dataset-Recommendation
│
├── app.py
├── src
│   ├── collectors
│   ├── recommender.py
│   └── ...
├── data
├── tests
├── requirements.txt
└── README.md
```

---

# 6. Features

| Feature | Description |
|----------|-------------|
| 📊 Dataset Recommendation | Recommend relevant datasets based on the project description |
| ⭐ Recommendation Score | Calculate and rank dataset relevance |
| 💡 Recommendation Reason | Explain why each dataset is recommended |
| 🤖 Recommended ML Algorithm | Suggest suitable machine learning algorithms |
| 📈 Evaluation Metric | Recommend appropriate evaluation metrics |
| 🔍 Dataset Comparison | Compare multiple datasets for informed selection |

---

# 7. Usage

## Demo

> Coming Soon

## Installation

```bash
git clone https://github.com/username/AI-Dataset-Recommendation.git

cd AI-Dataset-Recommendation

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Example

1. Enter a project description.
2. Click **Recommendation**.
3. Review the recommended datasets.
4. Compare datasets and select the most suitable one.

---

# 8. Limitations

현재 프로젝트의 한계는 다음과 같습니다.

- 데이터셋 출처가 UCI와 Kaggle로 제한되어 있습니다.
- 메타데이터 기반 추천으로 인해 추천 가능한 데이터셋 수가 제한적입니다.
- TF-IDF 기반 유사도는 의미적 문맥을 완전히 반영하지 못합니다.

---

# 9. Future Work

향후 개선 계획은 다음과 같습니다.

- OpenML 데이터셋 연동
- Hugging Face Datasets 연동
- Sentence-BERT 기반 의미 검색
- 추천 모델 고도화
- 사용자 피드백 기반 개인화 추천
- 다양한 AI 데이터 플랫폼 지원