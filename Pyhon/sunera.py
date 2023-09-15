import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift 
import matplotlib.pyplot as plt
import glob

folder_path = 'F:/7TH SEMESTER/EE405  Undergraduate Project I/kmz/panda/New folder' 
file_list = glob.glob(folder_path + "/*.csv")
data_frames = [pd.read_csv(file) for file in file_list]

df_list = []
for file in file_list:
    df = pd.read_csv(file)
    df_list.append(df)

df_concat = pd.concat(df_list)

x = df_concat['LONGITUDE']
y = df_concat['LATITUDE']