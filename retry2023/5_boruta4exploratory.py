import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from multiprocessing import cpu_count
import shap
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables


def objective(trial):
    # min_samples_split = trial.suggest_int("min_samples_split", 8, 16)
    # max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 64, 4))
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = trial.suggest_int('n_estimators', 100, 2000, 100)
    max_depth = trial.suggest_int('max_depth', 2, 7)  # , log=True)
    max_features = trial.suggest_categorical('max_features', [1.0, 'sqrt', 'log2'])

    clf = RandomForestClassifier(
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        # min_samples_split=min_samples_split,
        # max_leaf_nodes=max_leaf_nodes,
        criterion=criterion,
        random_state=0,
        n_jobs=int(cpu_count() * 2 / 3))

    clf.fit(X_train, Y_train)
    score = cross_val_score(clf, X_train, Y_train, n_jobs=int(cpu_count() * 2 / 3), cv=5).mean()

    return 1.0 - score  # accuracy_score(Y_test, clf.predict(X_test))
    # {'criterion': 'entropy', 'n_estimators': 1100, 'max_depth': 2, 'max_features': 'log2'}
    # 正答率:  0.5904761904761904


df = pd.read_table("/Volumes/Pegasus32R8/TTC/2023retry/df4grf_all_imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# 第２期のAQ素点からAQを計算
# AQの合計点を作成
df_AQ = df[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]]
print(df_AQ)

for i in ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]:
    df_AQ = df_AQ.replace({i: {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}})
    print(df_AQ)

for i in ["BB125", "BB126", "BB127", "BB132"]:
    df_AQ = df_AQ.replace({i: {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}})
    print(df_AQ)

df["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df["AQ_sum"])

# 本人または養育者が一方でも「1回以上あった」と回答した人をbulliedとする
df["bullied"] = 1
df["bullied"] = df["bullied"].where((df["AB61"] < 5) | (df["AD19"] < 5), 0)
# print(df[["bullied", "AB61", "AD19"]])

# 収入を500万円未満、1000万円未満、1000万円以上、で3つに分け直す
df_SES = df[["AB195"]]
print("SES素点", df_SES)
df_SES = df_SES.replace({i: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
                             6: 1, 7: 1, 8: 1, 9: 1, 10: 1,
                             11: 2}})
print("SES 3カテゴリー", df_SES)
df["SES"] = df_SES

df.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/df4grf_xty.csv")

df4boruta = df[df['OCS_0or1'] == 1]

# 1つでも「2回以上あった」の項目がある人をPLEありとする
df_ple = df4boruta[
    (df['CD57_1'] > 2) | (df['CD58_1'] > 2) | (df['CD59_1'] > 2) | (df['CD60_1'] > 2) | (df['CD61_1'] > 2) |
    (df['DD64_1'] > 2) | (df['DD65_1'] > 2) | (df['DD66_1'] > 2) | (df['DD67_1'] > 2) | (df['DD68_1'] > 2)
    ].copy()
print("PLEあり\n", df_ple)
df_ple["group"] = 1

# 全ての項目に回答があって、「1回あった」までの人はPLEなしとする
df_non = df4boruta[
    (df['CD57_1'] < 2) & (df['CD58_1'] < 2) & (df['CD59_1'] < 2) & (df['CD60_1'] < 2) & (df['CD61_1'] < 2) &
    (df['DD64_1'] < 2) & (df['DD65_1'] < 2) & (df['DD66_1'] < 2) & (df['DD67_1'] < 2) & (df['DD68_1'] < 2)
    ].copy()
print("PLEなし\n", df_non)
df_non["group"] = 0

df_concat = pd.concat([df_ple, df_non])

# 2期で強迫症状ありに絞って、PLE出現を予測する
print("2期に強迫症状あり\n", df_concat)
print("3期にPLEあり\n", df_concat["group"].sum())

y = df_concat["group"]
print(y)

X = df_concat.drop(['OCS_0or1', 'group',
                    "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1",
                    "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)
print(X.shape)
print(X.head())
# 参照 https://datadriven-rnd.com/2021-02-03-231858/

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=0.3, stratify=y, random_state=0)
print("Y_train", Y_train)
print("Y_test", Y_test)

"""
# ハイパーパラメータの自動最適化
study = optuna.create_study()
study.optimize(objective, n_trials=2000)

print(study.best_params)  # 求めたハイパーパラメータ
print("正答率: ", 1.0 - study.best_value)
# 'criterion': 'gini', 'n_estimators': 200, 'max_depth': 7, 'max_features': 1.0
# 正答率:  0.7642857142857142

optimised_rf = RandomForestClassifier(max_depth=study.best_params['max_depth'],
                                      max_features=study.best_params['max_features'],
                                      n_estimators=study.best_params['n_estimators'],
                                      random_state=0,
                                      n_jobs=int(cpu_count() * 2 / 3))

optimised_rf.fit(X_train, Y_train)
print(optimised_rf.classes_)
print(confusion_matrix(Y_test.values, optimised_rf.predict(X_test.values), labels=optimised_rf.classes_))
print("before boruta\n", accuracy_score(Y_test, optimised_rf.predict(X_test)))

# Borutaの実施
feat_selector = BorutaPy(optimised_rf,
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
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_test_selected = X_test.iloc[:, feat_selector.support_]

print('boruta後の変数の数:', X_train_selected.shape[1])
print('boruta後の変数:\n', X_train_selected.columns)

rf2 = RandomForestClassifier(
    max_depth=study.best_params['max_depth'],
    max_features=study.best_params['max_features'],
    n_estimators=study.best_params['n_estimators'],
    random_state=0,
    n_jobs=int(cpu_count() * 2 / 3)
)
rf2.fit(X_train_selected.values, Y_train.values)
"""

optimised_rf = RandomForestClassifier(
    max_depth=7,
    max_features=1.0,
    n_estimators=200,
    random_state=0,
    n_jobs=int(cpu_count() * 2 / 3)
)

optimised_rf.fit(X_train, Y_train)
print(optimised_rf.classes_)
print(confusion_matrix(Y_test.values, optimised_rf.predict(X_test.values), labels=optimised_rf.classes_))
print("before boruta\n", accuracy_score(Y_test, optimised_rf.predict(X_test)))

# Borutaの実施
feat_selector = BorutaPy(optimised_rf,
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2,
                         alpha=0.05,  # 有意水準
                         max_iter=100,  # 試行回数
                         perc=89,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
                         two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
                         random_state=0,
                         )

# データの二度漬けになるので特徴量選択する際にもtestを含めてはいけない
feat_selector.fit(X_train.values, Y_train.values)
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_test_selected = X_test.iloc[:, feat_selector.support_]

print('boruta後の変数の数:', X_train_selected.shape[1])
print('boruta後の変数:\n', X_train_selected.columns)

rf2 = RandomForestClassifier(
    max_depth=7,
    max_features=1.0,
    n_estimators=200,
    random_state=0,
    n_jobs=int(cpu_count() * 2 / 3)
)
rf2.fit(X_train_selected.values, Y_train.values)

print(rf2.classes_)
print(confusion_matrix(Y_test.values, rf2.predict(X_test_selected.values), labels=rf2.classes_))
print("after boruta\n", accuracy_score(Y_test, rf2.predict(X_test_selected)))
