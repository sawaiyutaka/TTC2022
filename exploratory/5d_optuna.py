import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
import shap
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from multiprocessing import cpu_count
from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables
import optuna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 2000)
    max_depth = trial.suggest_int('max_depth', 2, 8)  # , log=True)
    max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])

    clf = RandomForestClassifier(
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        n_jobs=int(cpu_count() / 2))

    score = cross_val_score(clf, X_train, Y_train, n_jobs=int(cpu_count() / 2), cv=5).mean()
    print('Accuracy: ', score)
    return score


df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/binary_3rd.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

y = df["group_3rd"]
print(y)

X = df.drop(["group_3rd"], axis=1)

# 参照！！：https://datadriven-rnd.com/2021-02-03-231858/
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
threshold_of_rate_of_same_value = 0.8  # (inner_fold_number - 1) / inner_fold_number
for col in X.columns:
    same_value_number = X[col].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / X.shape[0]))
del_var_num = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
X.drop(X.columns[del_var_num], axis=1, inplace=True)
"""
print(X.shape)
print(X.head())

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=0.2)
print("Y_train", Y_train)
print("Y_test", Y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10000)

trial = study.best_trial
print("Best hyperparameters: {}".format(trial.params))

optimised_rf = RandomForestClassifier(max_depth=study.best_params['max_depth'],
                                      max_features=study.best_params['max_features'],
                                      n_estimators=study.best_params['n_estimators'],
                                      n_jobs=int(cpu_count() / 2))

optimised_rf.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = optimised_rf.predict(X_test)
print(optimised_rf.classes_)
print(confusion_matrix(Y_test.values, optimised_rf.predict(X_test.values), labels=optimised_rf.classes_))
# 5分割交差検証による汎化性能の評価
scores = cross_val_score(optimised_rf, X_train, Y_train, n_jobs=int(cpu_count() / 2), cv=5)

# 評価結果の出力
print("5分割交差検証による汎化性能の評価: {}".format(100 * scores.mean()))

"""
# pパーセンタイルの最適化
corr_list = []
for n in range(10000):
    shadow_features = np.random.rand(X_train.shape[0]).T
    corr = np.corrcoef(X_train, shadow_features, rowvar=False)[-1]
    corr = abs(corr[corr < 0.95])
    corr_list.append(corr.max())

corr_array = np.array(corr_list)
perc = 100 * (1 - corr_array.max())
print('pパーセンタイル:', round(perc, 2))
"""

# Borutaの実施
feat_selector = BorutaPy(optimised_rf,
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2,
                         alpha=0.05,  # 有意水準
                         max_iter=100,  # 試行回数
                         perc=80,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
                         two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
                         random_state=0
                         )

# データの二度漬けになるので特徴量選択する際にもtestを含めてはいけない
feat_selector.fit(X_train.values, Y_train.values)
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_test_selected = X_test.iloc[:, feat_selector.support_]
print('boruta後の変数の数:', X_train_selected.shape[1])

rf2 = RandomForestClassifier(
    n_estimators=10000,
    random_state=42,
    n_jobs=int(cpu_count() / 2),
    max_features='sqrt'
)
rf2.fit(X_train_selected.values, Y_train.values)

print(rf2.classes_)
print(confusion_matrix(Y_test.values, rf2.predict(X_test_selected.values), labels=rf2.classes_))
print("after boruta\n", accuracy_score(Y_test, rf2.predict(X_test_selected)))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(rf2.predict, X_test_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test_selected)

shap.summary_plot(shap_values, max_display=len(X_test_selected.columns))
