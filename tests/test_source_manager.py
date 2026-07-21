from src.live_recommender import SourceManager


manager = SourceManager()

results = manager.search_all(
    project_description=(
        "house prices using income and location data"
    ),
    limit_per_source=3,
)

print(f"{len(results)} total results")

for dataset in results:
    print(
        dataset.source,
        dataset.dataset_name,
        dataset.popularity,
    )