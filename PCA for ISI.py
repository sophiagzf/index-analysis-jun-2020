#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('seaborn')

data1 = pd.read_excel('/Users/ganzhifei/Desktop/2020 SummerResearch/database/ISI.xlsx', sheet_name = "sheet1")
data1.columns = ['SgnMonth', 'DCEF_t0', 'RIPO_t1', 'IPON_t1', 'NA_t1', 'TURN_t0', 'CCI_t0', 'ISI']

names = ['DCEF_t0', 'RIPO_t1', 'IPON_t1', 'NA_t1', 'TURN_t0', 'CCI_t0']

# normalize

for kk in names:
    data1[kk] = (data1[kk] - data1[kk].mean())/data1[kk].std()

for name in names:

    if 't0' in name:
        new_name = name.replace('t0', 't1')
        data1.loc[:, new_name] = data1[name].shift(periods=1)
    elif 't1' in name:
        new_name = name.replace('t1', 't0')
        data1.loc[:, new_name] = data1[name].shift(periods=-1)

data1.dropna(inplace=True)

x1 = data1[['DCEF_t0', 'RIPO_t1', 'IPON_t1', 'NA_t1', 'TURN_t0',
       'CCI_t0', 'ISI', 'DCEF_t1', 'RIPO_t0', 'IPON_t0', 'NA_t0', 'TURN_t1',
       'CCI_t1']].values

pca = PCA(n_components=4)
pc = pca.fit_transform(x1)

print('Explained variance')
print(pca.explained_variance_)
print('Explained variance ratio')
print(pca.explained_variance_ratio_ * 100)
print(np.cumsum(pca.explained_variance_ratio_) * 100)

loadings = pca.components_.T@np.sqrt(pca.explained_variance_)
data1['sent_v0'] = pc@np.sqrt(pca.explained_variance_)

data1.corr()

corrdf = data1.corr()

res = []
for ii in ['DCEF', 'RIPO', 'IPON', 'NA', 'TURN', 'CCI']:
    t0 = ii + '_t0'
    t1 = ii + '_t1'
    corr0 = corrdf.loc['sent_v0', t0]
    corr1 = corrdf.loc['sent_v0', t1]
    if corr0 > corr1:
        res.append(t0)
    else:
        res.append(t1)

#for kk in res:
 #   data1[kk] = (data1[kk] - data1[kk].mean())/data1[kk].std()

# x2 = data1[res].values

# pca = PCA(n_components=4)
# pc = pca.fit_transform(x2)

# print('Explained variance')
# print(pca.explained_variance_)
# print('Explained variance ratio')
# print(pca.explained_variance_ratio_ * 100)
# print(np.cumsum(pca.explained_variance_ratio_) * 100)

# loadings = pca.components_.T@np.sqrt(pca.explained_variance_)
# data1['sent_v2'] = pc@np.sqrt(pca.explained_variance_)
# data1['sent_v2_norm'] = data1['sent_v2'] / data1['sent_v2'].std()

# regress on macro factors

PPI = pd.read_excel("/Users/ganzhifei/Desktop/2020 SummerResearch/database/PPI-累计同比.xlsx")
del PPI['序号']
PPI.columns = ("SgnMonth",'PPI')

CPI = pd.read_excel("/Users/ganzhifei/Desktop/2020 SummerResearch/database/CPI-累计同比.xlsx")
del CPI['序号']
CPI.columns = ("SgnMonth",'CPI')

MBCI = pd.read_excel("/Users/ganzhifei/Desktop/2020 SummerResearch/database/MBCI-一致指数.xlsx")
del MBCI['序号']
MBCI.columns = ("SgnMonth",'MBCI')

# how in merge is the way how we merge the data

data1 = data1.merge(CPI, on = 'SgnMonth', how = 'left').\
    merge(MBCI, on = 'SgnMonth', how = 'left').\
    merge(PPI, on = 'SgnMonth', how = 'left')
data1.dropna(inplace = True)

for kk in ['CPI', 'MBCI', 'PPI']:
    data1[kk] = (data1[kk] - data1[kk].mean())/data1[kk].std()

for ii in res:
    model = LinearRegression()
    model.fit(data1[['CPI', 'MBCI', 'PPI']], data1[ii])
    y_pred = model.predict(data1[['CPI', 'MBCI', 'PPI']])
    epsilon = data1[ii] - y_pred
    data1[ii + '_epsilon'] = epsilon

res2 = [ii + '_epsilon' for ii in res]
x2 = data1[res2].values

pca = PCA(n_components=3)
pc = pca.fit_transform(x2)

print('Explained variance')
print(pca.explained_variance_)
print('Explained variance ratio')
print(pca.explained_variance_ratio_ * 100)
print(np.cumsum(pca.explained_variance_ratio_) * 100)

loadings = pca.components_.T@np.sqrt(pca.explained_variance_)
data1['sent_v2'] = pc@np.sqrt(pca.explained_variance_)

data1 = data1.set_index(['SgnMonth'])
data1['sent_v2'].plot()
plt.show()

data1['sent_v2_norm'] = data1['sent_v2'] / data1['sent_v2'].std()