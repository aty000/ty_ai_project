AI Dataset Recommendation System
1. Project Overview
프로젝트 소개

AI Dataset Recommendation System은 AI 프로젝트에 적합한 데이터셋을 빠르게 탐색하고 추천하기 위한 데이터셋 추천 서비스이다.

사용자는 프로젝트에 대한 간단한 설명만 입력하면 UCI Machine Learning Repository와 Kaggle에서 관련 데이터셋을 검색하고, 프로젝트와의 관련성을 분석하여 적합한 데이터셋을 추천받을 수 있다.

또한 추천 결과와 함께 추천 이유, 적합한 머신러닝 알고리즘, 평가 지표를 제공하며 여러 데이터셋을 비교하여 프로젝트에 가장 적합한 데이터를 선택할 수 있도록 지원한다.

2. Motivation
프로젝트 개발 배경

AI 프로젝트를 시작할 때 가장 먼저 수행하는 작업은 프로젝트 목적에 적합한 데이터셋을 찾는 것이다.

그러나 대부분의 데이터셋은 여러 플랫폼에 분산되어 있으며 사용자는 각 사이트를 직접 검색하고 데이터셋의 설명과 특성을 비교해야 한다.

이 과정은 많은 시간이 소요될 뿐만 아니라 초보자에게는 어떤 데이터셋이 자신의 프로젝트에 적합한지 판단하기 어렵다.

본 프로젝트는 이러한 문제를 해결하기 위해 여러 데이터셋 플랫폼을 통합 검색하고 프로젝트 설명을 기반으로 적합한 데이터셋을 추천하는 시스템을 개발하였다.

3. Recommendation Process
Overall Workflow
Project Description
        │
        ▼
Keyword Extraction
        │
        ▼
Search UCI Repository
        │
        ├──────────┐
        ▼          │
Search Kaggle      │
        │          │
        └────┬─────┘
             ▼
 Candidate Datasets
             │
             ▼
 Metadata Collection
             │
             ▼
 TF-IDF Similarity
             │
             ▼
 Recommendation Score
             │
             ▼
 Top-N Recommendation

 Step 1. User Input

사용자는 프로젝트에 대한 간단한 설명을 입력한다.

예시

House Price Prediction
Medical Image Classification
Customer Churn Prediction

Step 2. Dataset Search

프로젝트 설명을 기반으로

UCI Machine Learning Repository
Kaggle

에서 관련 데이터셋을 검색한다.

Step 3. Metadata Collection

검색된 데이터셋의 메타데이터를 수집한다.

Dataset Name
Description
Task Type
Domain
Number of Instances
Number of Features
Popularity
Source

Step 4. Similarity Analysis

사용자 입력과 데이터셋 설명을 TF-IDF 기반으로 벡터화하고 Cosine Similarity를 이용하여 관련성을 계산한다.

Step 5. Recommendation Score

유사도와 메타데이터를 기반으로 최종 추천 점수를 계산하고 상위 데이터셋을 추천한다.

4. Tech Stack
| Category         | Technology        |
| ---------------- | ----------------- |
| Language         | Python            |
| Web Framework    | Streamlit         |
| Machine Learning | scikit-learn      |
| Data Processing  | Pandas            |
| Dataset Source   | UCI ML Repository |
| Dataset Source   | Kaggle API        |
| Version Control  | Git / GitHub      |

5. Project Structure
AI-Dataset-Recommendation
│
├── app.py
├── src
│   ├── collectors
│   │      ├── uci_collector.py
│   │      ├── kaggle_collector.py
│   │      └── common.py
│   │
│   ├── recommender.py
│   └── ...
│
├── data
│
├── tests
│
├── requirements.txt
│
└── README.md

6. Features
Dataset Recommendation

프로젝트 설명을 기반으로 관련 데이터셋 추천

Recommendation Score

추천 점수를 계산하여 가장 적합한 데이터셋 제공

Recommendation Reason

추천 이유 제공

Recommended Machine Learning Algorithm

추천 데이터셋에 적합한 머신러닝 알고리즘 제안

Recommended Evaluation Metric

프로젝트에 적합한 평가 지표 추천

Dataset Comparison

추천된 여러 데이터셋을 비교하여 선택 지원

비교 항목

Recommendation Score
Relevance Score
Task Match
Domain Match
Recommendation Reason
Recommended Algorithm
Evaluation Metric


7. Usage
Installation
git clone <repository>

cd AI-Dataset-Recommendation

pip install -r requirements.txt
Run
streamlit run app.py
Example
프로젝트 설명 입력
Recommendation 버튼 클릭
추천 결과 확인
데이터셋 비교


8. Limitations

현재 프로젝트의 한계

데이터셋 출처가 UCI와 Kaggle로 제한되어 있음
메타데이터 기반 추천으로 인해 추천 가능한 데이터셋 수가 제한적임
TF-IDF 기반 유사도는 의미적 문맥을 완전히 반영하지 못함


9. Future Work

향후 개선 계획

OpenML 데이터셋 연동
Hugging Face Datasets 연동
Sentence-BERT 기반 의미 검색
추천 모델 고도화
사용자 피드백 기반 개인화 추천
더 다양한 AI 데이터 플랫폼 지원