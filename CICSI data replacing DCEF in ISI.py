#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('seaborn')

data = pd.read_excel('/Users/ganzhifei/Desktop/2020 SummerResearch/database/CICSI.xlsx', sheet_name = "sheet1")
data1 = pd.read_excel('/Users/ganzhifei/Desktop/2020 SummerResearch/database/ISI.xlsx', sheet_name = "sheet1")

data.columns = ['SgnMonth', 'DCEF_t1', 'TURN_t0', 'IPON_t1', 'IPOR_t1', 'NIA_t0', 'CCI_t1', 'CICSI']

data['DCEF_t0'] = data1['上月封闭基金平均折价率']

names = ['DCEF_t0', 'TURN_t0', 'IPON_t1', 'IPOR_t1', 'NIA_t0', 'CCI_t1']

for name in names:

    if 't0' in name:
        new_name = name.replace('t0', 't1')
        data.loc[:, new_name] = data[name].shift(periods=1)
    elif 't1' in name:
        new_name = name.replace('t1', 't0')
        data.loc[:, new_name] = data[name].shift(periods=-1)

data.dropna(inplace=True)


x1 = data[['CCI_t1', 'TURN_t0', 'TURN_t1',
           'CCI_t0', 'IPON_t1', 'IPON_t0',
           'NIA_t1', 'NIA_t0', 'IPOR_t1', 'IPOR_t0',
           'DCEF_t1', 'DCEF_t0']].values

pca = PCA(n_components=4)
pc = pca.fit_transform(x1)

print('Explained variance')
print(pca.explained_variance_)
print('Explained variance ratio')
print(pca.explained_variance_ratio_ * 100)
print(np.cumsum(pca.explained_variance_ratio_) * 100)

loadings = pca.components_.T@np.sqrt(pca.explained_variance_)
data['sent_v0'] = pc@np.sqrt(pca.explained_variance_)

print(data.corr())


