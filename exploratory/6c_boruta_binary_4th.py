import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
import shap
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/binary_4th.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

y = df["group_4th"]
print(y)

X = df.drop(["group_4th"], axis=1)

# 参照！！：https://datadriven-rnd.com/2021-02-03-231858/

print(X.shape)
print(X.head())

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, stratify=y, random_state=0)

rf = RandomForestClassifier(
    n_estimators=2000,
    criterion='gini',
    n_jobs=int(cpu_count() / 2),
    max_depth=7,
    max_features=1.0,
    random_state=0
)
# 'criterion': 'gini', 'n_estimators': 2000, 'max_depth': 7, 'max_features': 1.0
rf.fit(X_train.values, Y_train.values)
print(rf.classes_)
print(confusion_matrix(Y_test.values, rf.predict(X_test.values), labels=rf.classes_))
print("before boruta\n", accuracy_score(Y_test, rf.predict(X_test)))

# Borutaの実施
feat_selector = BorutaPy(rf,
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2,
                         alpha=0.05,  # 有意水準
                         max_iter=100,  # 試行回数
                         perc=80,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
                         two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
                         random_state=0,
                         )

# データの二度漬けになるので特徴量選択する際にもtestを含めてはいけない
feat_selector.fit(X_train.values, Y_train.values)
X_selected = X.iloc[:, feat_selector.support_]
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_test_selected = X_test.iloc[:, feat_selector.support_]
print('boruta後の変数の数:', X_train_selected.shape[1])
print(X_train_selected.columns)

rf2 = RandomForestClassifier(
    n_estimators=2000,
    criterion='gini',
    n_jobs=int(cpu_count() / 2),
    # max_depth=7,
    max_features=1.0,
    random_state=0
)
rf2.fit(X_train_selected.values, Y_train.values)
print(rf2.classes_)
print(confusion_matrix(Y_test.values, rf2.predict(X_test_selected.values), labels=rf2.classes_))
print("after boruta\n", accuracy_score(Y_test, rf2.predict(X_test_selected)))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(rf2.predict, X_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_selected)

shap.plots.bar(shap_values, max_display=len(X_selected.columns))
shap.summary_plot(shap_values, max_display=len(X_selected.columns))