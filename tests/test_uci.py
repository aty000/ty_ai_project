from src.collectors.uci_collector import search_uci_datasets

results = search_uci_datasets(
    "house prices using income and location data",
    limit=5,
)

print(f"{len(results)} results")

for dataset in results:
    print(dataset.title, dataset.popularity)