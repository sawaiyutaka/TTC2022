import pprint
import sys
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, StratifiedKFold
from multiprocessing import cpu_count
import shap
from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/binary.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

y = df["group"]
print(y)

X = df.drop(["group"], axis=1)
print(X.shape)
print(X.head())

"""
# 分散が０の変数削除
del_num1 = np.where(X.var() == 0)
X = X.drop(X.columns[del_num1], axis=1)

# 変数選択（互いに相関関係にある変数の内一方を削除）
threshold_of_r = 0.95  # 変数選択するときの相関係数の絶対値の閾値
corr_var = search_highly_correlated_variables(X, threshold_of_r)
X.drop(X.columns[corr_var], axis=1, inplace=True)

# 同じ値を多くもつ変数の削除
inner_fold_number = 2  # CVでの分割数（予定）
rate_of_same_value = []
threshold_of_rate_of_same_value = 0.9  # (inner_fold_number - 1) / inner_fold_number
for col in X.columns:
    same_value_number = X[col].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / X.shape[0]))
del_var_num = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
X.drop(X.columns[del_var_num], axis=1, inplace=True)
"""
print('Original dataset shape %s' % Counter(y))

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# 参照！！：https://datadriven-rnd.com/2021-02-03-231858/
Y_train, Y_test, X_train, X_test = train_test_split(y_res, X_res, test_size=0.2, stratify=y_res, random_state=0)

# Apply RandomUnderSampler to undersample the majority class
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train, Y_train = rus.fit_resample(X_train, Y_train)
print("Yの0/1が同数になるようにアンダーサンプリング：", X_train.shape)
print(Y_train[Y_train == 1])
print(Y_train[Y_train == 0])

rf = RandomForestClassifier(
    n_estimators=1500,
    criterion="gini",
    max_depth=3,
    max_features='sqrt',
    n_jobs=int(cpu_count() / 2),
    random_state=0
)
# {'criterion': 'gini', 'n_estimators': 1500, 'max_depth': 3, 'max_features': 'sqrt'}
rf.fit(X_train.values, Y_train.values)
print(rf.classes_)
print(confusion_matrix(Y_test.values, rf.predict(X_test.values), labels=rf.classes_))
score = cross_val_score(rf, X_test, Y_test, n_jobs=int(cpu_count() / 2), cv=5).mean()
print("before boruta\n", score)

# cross-validation
kf = KFold(n_splits=10,
           shuffle=True,
           random_state=0)
skf = StratifiedKFold(n_splits=10,
                      shuffle=True,
                      random_state=0)
scoring = {'accuracy': make_scorer(accuracy_score),
           'kappa': make_scorer(cohen_kappa_score)}

scores_kf = cross_validate(rf,
                           X,  #
                           y,
                           cv=kf,
                           n_jobs=int(cpu_count() / 2),
                           scoring=scoring)
scores_skf = cross_validate(rf,
                            X,  #
                            y,
                            cv=skf,
                            n_jobs=int(cpu_count() / 2),
                            scoring=scoring)

pprint.pprint(scores_kf)
pprint.pprint(scores_skf)

plt.plot(scores_kf['test_accuracy'], label='accuracy_kf')
plt.plot(scores_skf['test_accuracy'], label='accuracy_skf')
plt.plot(scores_kf['test_kappa'], label='kappa_kf')
plt.plot(scores_skf['test_kappa'], label='kappa_skf')
plt.legend(loc="best")
plt.xlabel("#CV")
plt.ylabel("Index")
plt.show()

print("KFoldの正解率__________" + str(scores_kf['test_accuracy'].mean()))
print("StratifiedKFoldの正解率" + str(scores_skf['test_accuracy'].mean()))
print("KFoldのKappa__________" + str(scores_kf['test_kappa'].mean()))
print("StratifiedKFoldのKappa" + str(scores_skf['test_kappa'].mean()))
print("KFoldの正解率からStratifiedKFoldの正解率を引いた数値:" + str(
    scores_kf['test_accuracy'].mean() - scores_skf['test_accuracy'].mean()))
print("KFoldのKappaからStratifiedKFoldのKappaを引いた数値:" + str(
    scores_kf['test_kappa'].mean() - scores_skf['test_kappa'].mean()))

# Borutaの実施
feat_selector = BorutaPy(rf,
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2,
                         alpha=0.05,  # 有意水準
                         max_iter=100,  # 試行回数
                         perc=95,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
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

rf2 = rf
rf2.fit(X_train_selected.values, Y_train.values)
print(rf2.classes_)
print(confusion_matrix(Y_test.values, rf2.predict(X_test_selected.values), labels=rf2.classes_))
score2 = cross_val_score(rf2, X_test_selected, Y_test, n_jobs=int(cpu_count() / 2), cv=5).mean()
print("after boruta\n", score2)

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(rf2.predict, X_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_selected)

plt.figure()
shap.plots.bar(shap_values, max_display=len(X_selected.columns))
shap.summary_plot(shap_values, max_display=len(X_selected.columns))
