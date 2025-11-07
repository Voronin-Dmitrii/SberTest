import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

df = pd.read_csv("test_task_2_aggregated.csv", sep=";")

coords = np.radians(df[["location_latitude", "location_longitude"]].values)


def cluster_and_evaluate(eps_meters):
    eps_rad = eps_meters / 6371000
    db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords)
    df["cluster"] = db.labels_
    n_clusters = len(set(db.labels_))
    total_tickets = df.groupby("cluster")["num_of_tickets"].sum().sum()
    cluster_counts = df.groupby("cluster").size()
    max_points = cluster_counts.max()
    return n_clusters, total_tickets, max_points


eps_values = [10, 25, 50, 75, 100, 150, 200, 500, 1000]
results = [(eps, *cluster_and_evaluate(eps)) for eps in eps_values]

results_df = pd.DataFrame(results, columns=["eps_meters", "num_clusters", "total_tickets", "max_points"])
print(results_df)
