# Basic
# import scipy
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster import hierarchy as hc
from scipy.stats import spearmanr
# MlFlow
import mlflow
import mlflow.sklearn
# Ploting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Metrics
from sklearn.metrics import f1_score, fbeta_score, make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


st.title('Atrial Fibrilation Detector')

"""
	Using the dataset provided by the 2020 Physionet Challenge we've developed an Atrial Fibrilation Detector trained to
	identify AF diagnosed patiences from a dataset containing patients with different pathologies like: PAC, RBBB, I-AVB,
	PVC, LBBB, STD, STE and healthy individuals.

	Although data from 12-lead ECG was provided, for this first analysis we've only used the lead 2 data and we've processed
	the signals in order to create a dataframe consisting of features we believe will help us classify.
"""

@st.cache
def load_data(lead='lead2-HRV'):
	url = "https://storage.googleapis.com/kagglesdsdata/datasets/180/408/data.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201110%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201110T235109Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=4c664dd1cd351964a21752bbd2c65f5b5a29a689dbbc0988cbd81484f6b94ce483966b3a6ee37918c1e63cefc4737097004da24e1670c342fd67ea6437083c4661614adfc2db9aa9766f1f2495534ab9d4dc31852472a69c4900b9c7365200247bc0fe1b967d9980bb14955f7af8e7c45182ed95568c1e522378d7eb1acd864b0260e06cf5d4e4a3b81fa32129cdfc088688b8dac0b4bdc52a47aedb7b54dc0223ff40e984ab0e328f3b5b34014a3afee02a9798d3289c0c60f286fd523186fc48cfc0e338f7c6e7db33df0664373f42bb3b0f29a7dca90f8ab163218c62022e85f52555437796a56187ac337e01cb9522db22222ba584ac0b6acd30e8dde37b"
	df = pd.read_csv(url)
	return df

@st.cache
def filter_df(df_raw, q):
	df = df_raw.copy()
	cols = df_raw.columns
	cols = cols.drop('diagnosis')
	for col in cols:
	    df = df[df[col] < df[col].quantile(q)]
	df_raw = df.copy()
	return df_raw

df = load_data()
st.write(df.describe())
df_raw = df.drop(['Unnamed: 32', 'id'], axis=1).copy()

st.header("Dendrogram")
corr = np.round(spearmanr(df_raw.drop('diagnosis', axis=1)).correlation, 4)
corr_condensed = hc.distance.squareform(1 - corr)
z = hc.linkage(corr_condensed, method='average')
fig_den = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_raw.drop('diagnosis',axis=1).columns, orientation='left', leaf_font_size=16)
st.pyplot(fig_den, clear_figure=True)

"""
Select below to choose which features to drop.
"""
drop = st.multiselect('To drop', ['area_worst', 'area_mean', 'perimeter_mean'])
to_drop = drop
df_raw = df_raw.drop(to_drop, axis=1)


st.header("Correlation Matrix")
fig_cor = plt.figure(figsize=(16,10))
sns.heatmap(df_raw.corr())
st.pyplot(fig_cor, clear_figure=True)

st.header("Boxplots")
fig_box1 = plt.figure(figsize=(20,5))
p = sns.boxplot(data=df_raw)
plt.xticks(rotation=45)
st.pyplot(fig_box1, clear_figure=True)

"""
### Let's remove some outliers
Move the slider to keep everything below the Xth quantile
"""

q = st.slider("", 0.9, 1.0, 0.99, 0.01)
df_raw = filter_df(df_raw, q)

fig_box2 = plt.figure(figsize=(20,5))
sns.boxplot(data=df_raw)
plt.xticks(rotation=45)
st.pyplot(fig_box2, clear_figure=True)

# st.header("Pairplots")
# fig_pair = plt.figure(figsize=(20,17))
# sns.pairplot(data=df_raw.iloc[:,9:].sample(frac=0.1, random_state=42), hue='label', palette='Set2', height=1.5)
# st.pyplot(clear_figure=True)


st.header("Principal Component Analysis")
scal = StandardScaler()
df_scal = scal.fit_transform(df_raw.drop('diagnosis', axis=1))
n_comps = df_scal.shape[1]
pca = PCA(n_components = n_comps)
df_pca = pca.fit_transform(df_scal)

xpca = pd.DataFrame(df_pca)

sns.set_context("talk", font_scale=0.7)
plt.figure(figsize=(15,6))
plt.scatter(xpca.loc[(df_raw.diagnosis == 'M').ravel(),0],xpca.loc[(df_raw.diagnosis == 'M').ravel(),1], alpha = 0.3, label = 'M')
plt.scatter(xpca.loc[(df_raw.diagnosis == 'B').ravel(),0],xpca.loc[(df_raw.diagnosis == 'B').ravel(),1], alpha = 0.3, label = 'B')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Analysis before feature selection')
plt.legend(loc='upper right')
plt.tight_layout()
st.pyplot(clear_figure=True)



y = df_raw['diagnosis']
X = df_raw.drop('diagnosis', axis=1)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
res = pd.DataFrame({'model':[], 'f1':[]})

models = {
	'Logistic Regression': LogisticRegression(),
	'Random Forest': RandomForestClassifier(),
	'Suport Vectors': SVC(),
	'KN Neighbors': KNeighborsClassifier()}

for name, model in models.items():
	model.fit(X_train, y_train)
	f1 = f1_score(y_eval, model.predict(X_eval), pos_label='M')
	res = res.append({'model': f"{name}", 'f1': f1}, ignore_index=True)

st.write(res.sort_values('f1', ascending=False))
print(res.sort_values('f1', ascending=False))

"""
Random Forest Classifier seems to yield the best results.
"""

m = RandomForestClassifier()
m.fit(X_train, y_train)

preds = np.stack([t.predict(X_eval) for t in m.estimators_])

st.header("The Bootstrap Method")
n_iterations = 300
n_size = int(len(preds) * 0.60)

from sklearn.utils import resample
Lowers = []
Uppers = []
for i in range(len(preds[0])):
    
    means = []
    
    for _ in range(n_iterations):
        rs = resample(preds[:, i], n_samples=n_size, replace=True)
        means.append(np.mean(rs))
    
    alpha = 0.99
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(means, p))
    Lowers.append(lower)
    
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(means, p))
    Uppers.append(upper)
y = pd.Categorical(y_eval)
X = pd.DataFrame({'actuals': y.codes,
                  'preds': np.mean(preds, axis=0),
                  'std': np.std(preds, axis=0),
                  'var': np.var(preds, axis=0),
                  'upper': Uppers - np.mean(preds, axis=0),
                  'lower': np.mean(preds, axis=0) - Lowers
                 })
X.reset_index(inplace=True)
X = X.drop('index', axis=1)
print("Breast Data Model")
X[:10]

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(
	go.Bar(
	    name='Control',
	    y=X['preds'][:50],
	    error_y=dict(
	            type='data',
	            symmetric=False,
	            array=X['upper'][:50],
	            arrayminus=X['lower'][:50]
    	),
    )
)

fig.update_layout(shapes=[dict(type= 'line', yref='y', y0= 0.5, y1= 0.5, xref= 'x', x0= -1, x1= 50)])
fig.update_xaxes(automargin=True)
st.plotly_chart(fig)