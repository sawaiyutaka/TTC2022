import sys
import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt
from missingpy import MissForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
import shap
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from multiprocessing import cpu_count


df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/continuous.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

y = df["PLE_sum"]
print(y)

X = df.drop(["PLE_sum"], axis=1)

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
threshold_of_rate_of_same_value = (inner_fold_number - 1) / inner_fold_number
for col in X.columns:
    same_value_number = X[col].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / X.shape[0]))
del_var_num = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
X.drop(X.columns[del_var_num], axis=1, inplace=True)
"""
print(X.shape)
print(X.head())

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, random_state=42)

rf = RandomForestRegressor(max_depth=None,
                           max_features=1.0,
                           # X_train.shape[1],  # The number of features to consider when looking for the best split
                           # 'sqrt'も可能
                           min_samples_split=5,
                           min_samples_leaf=1,
                           n_estimators=5000,
                           n_jobs=int(cpu_count() / 2),
                           random_state=42)
rf.fit(X_train.values, Y_train.values)
# 学習済みモデルの評価
predicted_Y_val = rf.predict(X_test)
print("model_score: ", rf.score(X_test, Y_test))


feat_selector = BorutaPy(rf,
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2,
                         alpha=0.05,  # 有意水準
                         max_iter=100,  # 試行回数
                         perc=90,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
                         two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
                         random_state=0,
                         )
feat_selector.fit(X_train.values, Y_train.values)
print(X_train.columns[feat_selector.support_])
# 選択したFeatureを取り出し
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_test_selected = X_test.iloc[:, feat_selector.support_]
print(X_test_selected.head())
# 選択したFeatureで学習
rf2 = RandomForestRegressor(max_depth=None,
                            max_features=1.0,
                            # X_train.shape[1],  # The number of features to consider when looking for the best split
                            # 'sqrt'も可能
                            min_samples_split=5,
                            min_samples_leaf=1,
                            n_estimators=5000,
                            n_jobs=int(cpu_count() / 2),
                            random_state=42)
rf2.fit(X_train_selected.values, Y_train.values)
predicted_Y_val_selected = rf2.predict(X_test_selected.values)
print("model_score_2: ", rf2.score(X_test_selected, Y_test))

# Yの予測に重要なパラメータを探す
explainer = shap.Explainer(rf2.predict, X_test)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test_selected, max_evals=1200)
# max_evals=500 is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = 887!

shap.plots.bar(shap_values, max_display=len(X_test_selected.columns))
shap.summary_plot(shap_values, max_display=len(X_test_selected.columns))
