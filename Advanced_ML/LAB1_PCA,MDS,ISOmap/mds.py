import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm
from scipy.spatial import distance_matrix 

rawdata = pd.read_csv("zoo.data")
names = rawdata['name']
types = rawdata['type']
rawdata.drop(['name','type'], axis='columns', inplace=True)
#print("shape of 16 features:",feature_data.shape)
rawdata['feature13'] /= 2
#rawdata['feature7'] *= 2
scaler = MinMaxScaler()  
#rawdata_scaled = scaler.fit_transform(rawdata)
#rawdata_scaled['feature13'] *= 2
#rawdata_scaled['feature7'] *= 3
#distance_raw = distance_matrix(rawdata, rawdata)
#distance_scaledraw = scaler.fit_transform(distance_raw)
var_feature=rawdata.var()
distance = distance_matrix(rawdata, rawdata)
#df["col"] = 2 * df["col"]
#print(type(distance_scaled))
scale_mds = scaler.fit_transform(distance)
mds = MDS(2)
after_mds = mds.fit_transform(distance)
#feature13 ,7
#print('var_feature:\n', var_feature)

########## plotting from MDS ##################
#mds_plot = pd.DataFrame(data = after_mds)
mds_plot = pd.DataFrame(data = after_mds, columns = ['feature7','feature13'])
mds_plot['name'] = names
#print('mds_plot:\n',mds_plot)
plt.clf()
plt.title("MDS_2D")
mds_list = mds_plot.values.tolist()
#print('mds_list:\n',mds_list)

for x,y,z in mds_list:
    plt.annotate(z, xy=(float(x),float(y)), size=10)
eucs = [x for (x,y,z) in mds_list]
covers = [y for (x,y,z) in mds_list]

mds_plot['type'] = types
mds_list = mds_plot.values.tolist()
colors = cm.rainbow(np.linspace(0, 1, len(types)))
groups = mds_plot.groupby('type')
for name, group in groups:
    plt.plot(group["feature7"], group["feature13"], marker="o", linestyle="", label=name)
plt.legend()
plt.savefig("MDS_2D.png")      
