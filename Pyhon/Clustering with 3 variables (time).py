import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift 
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import glob
import folium
from datetime import datetime
from sklearn.preprocessing import StandardScaler


#Retrieve the file paths of all the CSV files in the specified folder
folder_path = 'F:/7TH SEMESTER/EE405  Undergraduate Project I/Excel Files'
file_list = glob.glob(folder_path + "/*.csv")

# Read data from CSV files
data_frames = [pd.read_csv(file) for file in file_list]

# Set latitude and longitude ranges
lat_min = 7.244728
lat_max = 7.265915
lon_min = 80.590522
lon_max = 80.604884

# Filter rows based on latitude and longitude ranges
for i, df in enumerate(data_frames):
    data_frames[i] = df[(df['LATITUDE'] >= lat_min) & (df['LATITUDE'] <= lat_max) & (df['LONGITUDE'] >= lon_min) & (df['LONGITUDE'] <= lon_max)]


df_concat = pd.concat(data_frames)

x = df_concat['LONGITUDE']
y = df_concat['LATITUDE']
z = df_concat['TIME']

data = np.column_stack([x,y,z])
print(data)

#scale the data before using the Algorithm
scaler = StandardScaler()
X = scaler.fit_transform(data)

#------------------------------------OUT LIER FACTOR
# Outlier detection using LOF
lof = LocalOutlierFactor()
outlier_scores = lof.fit_predict(X)
outliers = X[outlier_scores == -1]  # Extract outlier points

# Remove outliers from the dataset
X = X[outlier_scores != -1]
#-------------------------------------

# Visualize the data points
plt.scatter(X[:,0], X[:,1])
plt.show()

#Perform Mean Shift clustering on the data
bandwidth = 0.8 # Adjust the bandwidth(meanshift radius) parameter as desired
cluster_all = False  # Set cluster_all to False to ignore noise points ,cluster_all=cluster_all
ms = MeanShift(bandwidth = bandwidth,cluster_all=cluster_all)
#ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

 # Determine the number of clusters
n_clusters_ = len(np.unique(labels))
#print(f"Number of estimated clusters in {csv_file}: {n_clusters_}")

# Define a list of colors to use for the cluster labels
colors = 100 * ['r.','g.','b.','c.','k.','y.','m.']

    # Visualize the clustered data points
for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=5)

plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()

#rescale cluster centers to the original scale

rescaled_centroid = scaler.inverse_transform(cluster_centers)

# Create a map centered at a specific location

map = folium.Map(location=[7.265841,80.5777642], zoom_start=13)
list=[]
for row in rescaled_centroid:
    new_row = [row[1],row[0]]
    list.append(new_row)

for point in list:
    folium.Marker(point).add_to(map)

map.save("test.html")

