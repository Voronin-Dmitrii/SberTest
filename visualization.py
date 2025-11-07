from shapely.geometry import MultiPoint, Polygon, LineString, Point
import folium
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import matplotlib.colors as colors

df = pd.read_csv("test_task_2_aggregated.csv", sep=";")

coords = np.radians(df[["location_latitude", "location_longitude"]].values)

# 200м кластеризация
eps_meters = 200
eps_rad = eps_meters / 6371000
db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords)
df["cluster"] = db.labels_

# позиция камеры
center_lat = df["location_latitude"].mean()
center_lon = df["location_longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# цвета кластеров
n_clusters = len(set(df["cluster"]))
colormap = cm.rainbow(np.linspace(0, 1, n_clusters))
rgb2hex = lambda rgb: colors.rgb2hex(rgb)

# точки
for _, row in df.iterrows():
    cluster_color = rgb2hex(colormap[int(row["cluster"]) % len(colormap)])
    folium.CircleMarker(
        location=[row["location_latitude"], row["location_longitude"]],
        radius=4,
        color=cluster_color,
        fill=True,
        fill_color=cluster_color,
        fill_opacity=0.8,
        popup=f"ATM ID: {int(row['atm_id'])} \nTickets: {int(row['num_of_tickets'])} \nCluster: {int(row['cluster'])}",
    ).add_to(m)

# обводка кластеров
for cluster_id in df["cluster"].unique():
    cluster_points = df[df["cluster"] == cluster_id][["location_latitude", "location_longitude"]].values
    hull = MultiPoint(cluster_points).convex_hull
    if isinstance(hull, Polygon):
        folium.Polygon(
            locations=[list(coord) for coord in hull.exterior.coords],
            color=rgb2hex(colormap[int(cluster_id) % len(colormap)]),
            weight=2,
            fill=True,
            fill_opacity=0.2
        ).add_to(m)
    elif isinstance(hull, LineString):
        folium.PolyLine(
            locations=[list(coord) for coord in hull.coords],
            color=rgb2hex(colormap[int(cluster_id) % len(colormap)]),
            weight=2
        ).add_to(m)

# подпись кластеров
cluster_summary = df.groupby("cluster").agg({
    "location_latitude": "mean",
    "location_longitude": "mean",
    "num_of_tickets": "sum",
    "atm_id": "count"
}).reset_index().rename(columns={"atm_id": "atm_count"})
for _, row in cluster_summary.iterrows():
    folium.map.Marker(
        [row["location_latitude"], row["location_longitude"]],
        icon=folium.DivIcon(
            html=f"""
                <div style="font-size: 8pt; font-weight: bold; color: black;
                            background-color: rgba(255,255,255,0.8);
                            border-radius: 6px; padding: 4px; text-align: center;
                            ">
                    Кластер: {int(row['cluster'])}<br>
                    Банкоматов: {int(row['atm_count'])}<br>
                    Заявок: {int(row['num_of_tickets'])}
                </div>
            """
        )
    ).add_to(m)

m.save("clusters_200m_map.html")
