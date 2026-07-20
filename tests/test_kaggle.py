from src.collectors.kaggle_collector import search_kaggle_datasets

results = search_kaggle_datasets(
    "house prices using income and location data",
    limit=5,
)

print(f"{len(results)} results")

for dataset in results:
    print(dataset.title)