import pandas as pd

df = pd.read_csv("test_task_2.csv", sep=";")
df["location_latitude"] = df["location_latitude"].str.replace(",", ".").astype(float)
df["location_longitude"] = df["location_longitude"].str.replace(",", ".").astype(float)

# агрегация банкоматов по atm_id
tickets_sum = df.groupby("atm_id")["num_of_tickets"].sum().reset_index()

idx_max_tickets = df.groupby("atm_id")["num_of_tickets"].idxmax()
df_max = df.loc[idx_max_tickets, ["atm_id", "location_latitude", "location_longitude"]].reset_index(drop=True)

df_agg = df_max.merge(tickets_sum, on="atm_id")

df_agg.to_csv("test_task_2_aggregated.csv", sep=";", index=False)
