#name,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,feature11,feature12,feature13,feature14,feature15,feature16,type
#18 columns in total

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm

rawdata = pd.read_csv("zoo.data")
#pxrint("rawdata.shape: ", rawdata.shape)
#print("rawdata:\n", rawdata)
names = rawdata['name']
types = rawdata['type']
#print('names:\n',names)
#raw_plot = rawdata.drop(['name'])
rawdata.drop(['name','type'], axis='columns', inplace=True)
#print("shape of 16 features:",feature_data.shape)

#feature13 is the number of legs(0:a kind of worm or fish),
#rest is boolean
pca = PCA(n_components=2)

#x_std = StandardScaler().fit_transform(x)
#pca.fit_transform(x_std)
scale_data = StandardScaler().fit_transform(rawdata)
principalComponents = pca.fit_transform(scale_data)

principalDF = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDF['name'] = names
#principalDF['type'] = types
#print('principlaDF:\n',principalDF)

plt.clf()
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.title("PCA_two_components")

DF_list = principalDF.values.tolist()
#print('list:',DF_list)

for x,y,z in DF_list:
    plt.annotate(z, xy=(float(x),float(y)), size=10)

eucs = [x for (x,y,z) in DF_list]
covers = [y for (x,y,z) in DF_list]
#p1 = plt.plot(eucs,covers,color="black", alpha=0.5)

principalDF['type'] = types
DF_list = principalDF.values.tolist()

colors = cm.rainbow(np.linspace(0, 1, len(types)))

groups = principalDF.groupby('type')
for name, group in groups:
    plt.plot(group["principal component 1"], group["principal component 2"], marker="o", linestyle="", label=name)
plt.legend()
plt.savefig("PCA_two_component.png")
