from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from boruta import BorutaPy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2023retry/df4grf_all_imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# social cohesion
df_social_cohesion = df[["AA57", "AA58", "AA59", "AA60", "AA61"]]
print(df_social_cohesion)
df["social_cohesion"] = df_social_cohesion.sum(axis=1)
print("social cohesion sum\n", df["social_cohesion"])

# atopy
df["atopy"] = 1
df["atopy"] = df["atopy"].where((df["AF37"] == 1) | (df["AF38"] == 1), 0)  # Falseのとき、NaNのかわりに値を代入
print("atopy\n", df[["atopy", "AF37", "AF38"]])

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

df = df.drop({"BB39", "BB83", 'OCS_sum',  # 第２期強迫
              "AB195", "AB61", "AD19",  # 第１期SES、第１期いじめられ
              "AD57", "AD58", "AD59", "AD60", "AD61",  # 第１期PLE
              "BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132",  # AQ
              "AA57", "AA58", "AA59", "AA60", "AA61",  # social cohesion
              "AF37", "AF38",  # atopy
              }, axis=1)

df.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/df4grf_xty.csv")

df4boruta = df[df['OCS_0or1'] == 1]  # ここをdfのみとすると、強迫の有無に関わらず解析する

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
df_concat.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/ocs2ple.csv")
y = df_concat["group"]
print(y)

X = df_concat.drop(['OCS_0or1', 'group',
                    "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1",
                    "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)
print("人数とXの変数", X.shape)
print(X.head())


# 参照 https://datadriven-rnd.com/2021-02-03-231858/

def youden_index_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    youden_index = sensitivity + specificity - 1
    return youden_index


# 外側ループのCVの分割設定
outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# ハイパーパラメータの探索空間
param_grid = {
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # list(range(101)),
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [1, 2],
    'random_state': [42],
    'n_jobs': [int(cpu_count() / 2)]
}

# Youden Indexを最大化するためのスコア関数を作成
scoring = make_scorer(youden_index_score, greater_is_better=True)

# rNCVでのハイパーパラメータの探索
best_score = 0
best_params = {}

repeats = 100
result_list = []
for _ in range(repeats):

    for train_index, val_index in outer_cv.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 内側ループのCVの分割設定
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True)

        # GridSearchCVを使用して最適なハイパーパラメータを見つける
        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv)
        grid_search.fit(X_train, y_train)

        # 最適なハイパーパラメータを取得
        params = grid_search.best_params_
        score = grid_search.best_score_

        print("score: ", score)
        print("params: ", params)

        # データの二度漬けになるので特徴量選択する際にもtestを含めてはいけない
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Borutaアルゴリズムによる特徴量選択
        boruta_selector = BorutaPy(
            model,
            n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
            verbose=2,
            alpha=0.05,  # 有意水準
            max_iter=100,  # 試行回数
            perc=95,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
            two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
            random_state=0
        )
        boruta_selector.fit(np.array(X_train), np.array(y_train))

        # 有効な特徴量の選択
        selected_features = X.columns[boruta_selector.support_].tolist()
        # 前回までに選ばれた要素のリストを取得
        previous_items = result_list[-1] if result_list else []

        # 新しい行のデータを作成
        new_row = [item if item in selected_features else "" for item in previous_items]

        # 初めて選ばれた要素を新しい行に追加
        for item in selected_features:
            if item not in previous_items:
                new_row.append(item)

        # 新しい行を結果のリストに追加
        result_list.append(new_row)
        X_selected = X[selected_features]

        print('boruta後の変数の数:', X_selected.shape[1])
        print(X_selected.columns)

print(result_list)
# 結果をDataFrameに変換
result_df = pd.DataFrame(result_list)

# 結果をCSVファイルとして保存
result_df.to_csv('/Volumes/Pegasus32R8/TTC/2023retry/X_selected_list.csv')
