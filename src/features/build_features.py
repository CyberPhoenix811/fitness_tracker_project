import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


#load the data

df = pd.read_pickle("../../src/features/outliers_removed_chauvenet.pkl")
predictors = df.columns[:6]

plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams["figure.dpi"] = 100

#Handling the missing values

for col in predictors:
    df[col] = df[col].interpolate()
    
#Counting duration of the sets

for s in np.sort(df["Set"].unique()):
    start = df[df["Set"] == s].index[0]
    stop = df[df["Set"] == s].index[-1]
    duration = stop - start
    df.loc[(df["Set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["Category"])["duration"].mean()

duration_df.iloc[0]/5
duration_df.iloc[1]/10

# Butterworth Lowpass Filter
df_lowpass = df.copy()
lowpass = LowPassFilter()

fs = 1000/200
cutoff = 1.2

df_lowpass = lowpass.low_pass_filter(df_lowpass,"acc_y",fs, cutoff, order=5)
subset = df_lowpass[df_lowpass["Set"] == 15]


fig, ax = plt.subplots(nrows= 2, sharex= True, figsize = (20,10))
ax[0].plot(subset["acc_y"].reset_index(drop = True), label = "raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop = True), label = "lowpass filter data")

ax[0].legend(
    loc = "upper center", ncol = 3 , bbox_to_anchor = (0.5,1.15), fancybox = True, shadow = True
)
ax[1].legend(
    loc = "upper center", ncol = 3 , bbox_to_anchor = (0.5,1.15), fancybox = True, shadow = True
)
ax[1].set_xlabel('Samples')

for col in predictors:
    df_lowpass = lowpass.low_pass_filter(df,col=col,sampling_frequency=fs, cutoff_frequency=cutoff, order= 5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    
# Principle Component Analysis
df_pca = df_lowpass.copy()
pca = PrincipalComponentAnalysis()

pca_values = pca.determine_pc_explained_variance(df_pca, predictors)
plt.figure(figsize=(10,10))
plt.plot(range(1, len(predictors)+ 1), pca_values)
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance")
plt.legend

df_pca = pca.apply_pca(df_pca, predictors, number_comp= 3)
subset1 = df_pca[df_pca["Set"] == 45]
subset1[["pca_1", "pca_2", "pca_3"]].plot()

# Sum of Squares

df_squared = df_pca.copy()
acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset1 = df_squared[df_squared["Set"] == 15]
subset1[["acc_r", "gyr_r"]].plot(subplots= True)

# Temporal Abstraction

df_temporal = df_squared.copy()
numab = NumericalAbstraction()

predictors1 =  ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',"acc_r", "gyr_r"]

ws = int(1000/200)

for col in predictors1:
    df_temporal = numab.abstract_numerical(df_temporal, [col], ws, aggregation_function="mean")
    df_temporal = numab.abstract_numerical(df_temporal, [col], ws, aggregation_function="std")
    
df_temporal_list = []
for s in df_temporal["Set"].unique():
    subset = df_temporal[df_temporal["Set"] == s].copy()
    for col in predictors1:
        subset = numab.abstract_numerical(subset, [col], ws, aggregation_function="mean")
        subset = numab.abstract_numerical(subset, [col], ws, aggregation_function="std")
    df_temporal_list.append(subset)
    
df_temporal = pd.concat(df_temporal_list)

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()

# Fourier transformation (Frequency Abstraction)

df_freq = df_temporal.copy().reset_index()
fourier = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200) # window size is the average duration of a set i.e 2.8 sec(2800 ms) and interval is 200 ms

df_freq_list = []
for s in df_freq["Set"].unique():
    subset = df_freq[df_freq["Set"] == s].copy().reset_index(drop = True)
    subset = fourier.abstract_frequency(subset, predictors1, ws, fs)
    print("Fourier Transformation completed for set ", s)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop = True)    

# Handling the overlapping windows

df_freq = df_freq.dropna()

df_freq = df_freq[::2]


# Clustering the data points

df_cluster = df_freq.copy()
cluster_col = ['acc_x', 'acc_y', 'acc_z']
values = range(2,10)
inertias = []
for k in values:
    subset = df_cluster[cluster_col]
    Kmeans = KMeans(n_clusters = k, n_init = 20, random_state = 0)
    cluster_labels = Kmeans.fit_predict(subset)
    inertias.append(Kmeans.inertia_)
    
plt.figure(figsize=(10,10))
plt.plot(values, inertias)
plt.xlabel(" k values")
plt.ylabel("Sum of squared distances")
plt.show()

# we have choosen the k value of 4 or 5 as they show the best no. of cluster to form in the data
subset = df_cluster[cluster_col]
Kmeans = KMeans(n_clusters = 4, n_init = 20, random_state = 0)
df_cluster["clusters"] = Kmeans.fit_predict(subset)

# Plot the cluster points

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

# Plot cluster points
for c in df_cluster["clusters"].unique():
    subset = df_cluster[df_cluster["clusters"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label = c)

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Clustering')

plt.legend()
plt.show()

# Export the dataset
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")