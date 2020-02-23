import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv("breast-cancer-wisconsin.data.txt")

df.drop("id", axis=1,inplace=True)
# print(df.head())
df.replace("?", -999, inplace=True)
fulldata = df.astype(int).values.tolist()
data = {2:[],4:[]}

for point in fulldata:
    data[point[-1]].append(point[0:-1])

pred_pt = [1,3,1,2,5,5,2,3,1]
k = 23
distances = []

for key in data:
    for pt in data[key]:
        dist = np.linalg.norm(np.array(pt) - np.array(pred_pt))
        distances.append((dist, key))

votes = []
for i in sorted(distances)[:k]:
    votes.append(i[1])

print(Counter(votes).most_common(1)[0][0])