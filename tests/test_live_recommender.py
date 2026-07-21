from src.live_recommender import LiveRecommender


recommender = LiveRecommender()

results = recommender.recommend(
    project_description=(
        "house prices using income and location data"
    ),
    limit=5,
    limit_per_source=10,
)

print(f"{len(results)} recommendations")

for rank, dataset in enumerate(
    results,
    start=1,
):
    print(
        rank,
        dataset.source,
        dataset.dataset_name,
        f"task={dataset.task_type}",
        f"domain={dataset.domain}",
        f"confidence={dataset.task_confidence:.0%}",
        f"format={dataset.data_format}",
        f"algorithms={dataset.recommended_algorithms}",
        f"metrics={dataset.recommended_metrics}",
        f"signals={dataset.analysis_signals}",
        f"retrieval={dataset.retrieval_score:.3f}",
        f"relevance={dataset.relevance_score:.3f}",
        f"popularity={dataset.popularity:.3f}",
        f"final={dataset.final_score:.3f}",
    )