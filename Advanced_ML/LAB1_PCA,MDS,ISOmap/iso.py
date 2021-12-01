#name,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,feature12,feature13,feature14,feature15,feature16,type
#18 columns in total

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm
from scipy.spatial import distance_matrix

rawdata = pd.read_csv("zoo.data")
#print("rawdata:\n", rawdata)
names = rawdata['name']
types = rawdata['type']
rawdata.drop(['name','type'], axis='columns', inplace=True)
scale_data = StandardScaler().fit_transform(rawdata)

#X, color = scale_data
####### Isomap #######
#fig = plt.figure(figsize=(15, 8))

Y = Isomap(n_neighbors=10, n_components=2).fit_transform(scale_data)
iso_plot = pd.DataFrame(data = Y, columns = ['component1','component2'])

iso_plot['name'] = names
#print('iso_plot:\n',iso_plot)
plt.clf()
plt.xlabel("component1")
plt.ylabel("component2")
plt.title("Isomap_2D")
iso_list = iso_plot.values.tolist()

for x,y,z in iso_list:
    plt.annotate(z, xy=(float(x),float(y)), size=10)
eucs = [x for (x,y,z) in iso_list]
covers = [y for (x,y,z) in iso_list]
#p1 = plt.plot(eucs,covers,color="black", alpha=0.5)
iso_plot['type'] = types
iso_list = iso_plot.values.tolist()
colors = cm.rainbow(np.linspace(0, 1, len(types)))
groups = iso_plot.groupby('type')
for name, group in groups:
    plt.plot(group["component1"], group["component2"], marker="o", linestyle="", label=name)
plt.legend()
plt.savefig("ISO_2D_k=10.png")
