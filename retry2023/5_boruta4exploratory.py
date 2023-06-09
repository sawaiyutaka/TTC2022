import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, f1_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from multiprocessing import cpu_count

from dcekit.variable_selection import search_highly_correlated_variables

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
df_SES = df_SES.replace({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 2})
print("SES 3カテゴリー", df_SES)
df["SES"] = df_SES

df = df.drop(["BB39", "BB83", 'OCS_sum',  # 第２期強迫
              "AB195", "AB61", "AD19",  # 第１期SES、第１期いじめられ
              "AD57", "AD58", "AD59", "AD60", "AD61",  # 第１期PLE
              "BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132",  # AQ
              ], axis=1)

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
print("3期・4期にPLEあり\n", df_concat["group"].sum())

y = df_concat["group"]
print(y)

X = df_concat.drop(['OCS_0or1', 'group',
                    "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1",
                    "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)
print("相関係数で変数選択する前", X.shape)
print(X.head())
# 参照 https://datadriven-rnd.com/2021-02-03-231858/

# 変数選択（互いに相関関係にある変数の内一方を削除）
threshold_of_r = 0.7  # 変数選択するときの相関係数の絶対値の閾値
corr_var = search_highly_correlated_variables(X, threshold_of_r)
X.drop(X.columns[corr_var], axis=1, inplace=True)
print("相関係数で変数選択した後", X.shape)

"""
# 同じ値を多くもつ変数の削除
inner_fold_number = 5  # CVでの分割数（予定）
rate_of_same_value = []
threshold_of_rate_of_same_value = (inner_fold_number - 1) / inner_fold_number
for col in X.columns:
    same_value_number = X[col].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / X.shape[0]))
del_var_num = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
X.drop(X.columns[del_var_num], axis=1, inplace=True)
print("同じ値かで変数選択した後", X.shape)
"""

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=0.3, stratify=y, random_state=0)
print("Y_train", Y_train)
print("Y_test", Y_test)

# ハイパーパラメータの候補を指定します
param_grid = {
    'n_estimators': list(range(1000, 2001, 200)),
    'max_depth': list(range(4, 8)),
    'max_features': ['sqrt'],
    'criterion': ['gini', 'entropy'],
    'random_state': [0]
}

# スコアリング指標として設定
scorer = make_scorer(recall_score)

# Random Forestモデルを作成します
rf_model = RandomForestClassifier()

# GridSearchCVを作成し、トレーニングデータでハイパーパラメータの最適化を行います
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=int(cpu_count() * 5 / 6))
grid_search.fit(X_train, Y_train)
results = pd.DataFrame(grid_search.cv_results_)
print("交差検証の各ハイパーパラメータ組み合わせに対するスコアやその他の評価指標\n", results)

# 最適なハイパーパラメータの結果の表示
print("Best parameters:", grid_search.best_params_)
print("Best recall_score:", grid_search.best_score_)

# グリッドサーチの全ての結果の表示
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print("recall_score:", mean_score, "Parameters:", params)

optimised_rf = RandomForestClassifier(max_depth=grid_search.best_params_['max_depth'],
                                      max_features=grid_search.best_params_['max_features'],
                                      n_estimators=grid_search.best_params_['n_estimators'],
                                      criterion=grid_search.best_params_['criterion'],
                                      random_state=0,
                                      n_jobs=int(cpu_count() * 5 / 6))

optimised_rf.fit(X_train, Y_train)
print(optimised_rf.classes_)
print(confusion_matrix(Y_test.values, optimised_rf.predict(X_test.values), labels=optimised_rf.classes_))
print("accuracy score before boruta\n", accuracy_score(Y_test, optimised_rf.predict(X_test)))

# Borutaの実施
feat_selector = BorutaPy(optimised_rf,
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2,
                         alpha=0.05,  # 有意水準
                         max_iter=100,  # 試行回数
                         perc=100,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
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
    max_depth=grid_search.best_params_['max_depth'],
    max_features=grid_search.best_params_['max_features'],
    n_estimators=grid_search.best_params_['n_estimators'],
    criterion=grid_search.best_params_['criterion'],
    random_state=0,
    n_jobs=int(cpu_count() * 5 / 6)
)
rf2.fit(X_train_selected.values, Y_train.values)

"""
optimised_rf = RandomForestClassifier(
    max_depth=4,
    max_features=1.0,
    n_estimators=400,
    random_state=0,
    n_jobs=int(cpu_count() * 5 / 6)
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
                         perc=100,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
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
    max_depth=4,
    max_features=1.0,
    n_estimators=400,
    random_state=0,
    n_jobs=int(cpu_count() * 5 / 6)
)
rf2.fit(X_train_selected.values, Y_train.values)
"""
print(rf2.classes_)
print(confusion_matrix(Y_test.values, rf2.predict(X_test_selected.values), labels=rf2.classes_))
print("accuracy score after boruta\n", accuracy_score(Y_test, rf2.predict(X_test_selected)))
