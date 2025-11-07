import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

df = pd.read_csv("test_task_2_aggregated.csv", sep=";")

coords = np.radians(df[["location_latitude", "location_longitude"]].values)

eps_meters = 200
eps_rad = eps_meters / 6371000
db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords)

df["cluster"] = db.labels_

cluster_summary = (
    df.groupby("cluster")
    .agg(
        atm_count=("atm_id", "count"),
        total_tickets=("num_of_tickets", "sum"),
        mean_lat=("location_latitude", "mean"),
        mean_lon=("location_longitude", "mean")
    )
    .reset_index()
)

cluster_summary.to_csv("clusters_summary.csv", index=False) #Информация по кластерам
df.to_csv("test_task_2_with_clusters.csv", sep=";", index=False) #Банкоматы с № кластера

