import pandas as pd
import json
import numpy as np
from sklearn.neighbors import BallTree

# load sensor 1 and sensor 2 data
sensor1_df = pd.read_csv("SensorData1.csv")

with open("SensorData2.json", "r") as f:
    sensor2_data = json.load(f)
sensor2_df = pd.DataFrame(sensor2_data)

# convert latitude and longitude from degrees to radians since the haversine formula expects radians
sensor1_coords = np.deg2rad(sensor1_df[['latitude', 'longitude']].to_numpy())
sensor2_coords = np.deg2rad(sensor2_df[['latitude', 'longitude']].to_numpy())

# build a BallTree for sensor2 coordinates using the haversine metric.
tree = BallTree(sensor2_coords, metric='haversine')

# 100 meter search radius to radians
radius = 100 / 6371000

# query the ballTree for each sensor1 coordinate to find sensor2 points within the radius.
results = tree.query_radius(sensor1_coords, r=radius)

# gather the matching pairs ids (Sensor1_id, Sensor2_id)
matches = []
for i, sensor2_indices in enumerate(results):
    sensor1_id = sensor1_df.iloc[i]['id']
    for j in sensor2_indices:
        sensor2_id = sensor2_df.iloc[j]['id']
        matches.append((sensor1_id, sensor2_id))

# printing output
print("Matches (Sensor1 ids, Sensor2 ids):")
for pair in matches:
    print(pair)