from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from boruta import BorutaPy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

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

# 外側ループのCVの分割設定
outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# ハイパーパラメータの探索空間
param_grid = {
    'n_estimators': list(range(101)),
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [1, 2],
    'random_state': [42],
    'n_jobs': [int(cpu_count() / 2)]
}

# rNCVでのハイパーパラメータの探索
best_score = 0
best_params = {}
for train_index, val_index in outer_cv.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 内側ループのCVの分割設定
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # GridSearchCVを使用して最適なハイパーパラメータを見つける
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(),
        param_grid=param_grid,
        cv=inner_cv)
    grid_search.fit(X_train, y_train)

    # 最適なハイパーパラメータを取得
    params = grid_search.best_params_
    score = grid_search.best_score_

    # 最も性能の良いモデルを保存
    if score > best_score:
        best_score = score
        best_params = params

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

print("best_score: ", best_score)
print("best_params: ", best_params)

# 最終的な分類器の訓練
model = xgb.XGBClassifier(**best_params)
model.fit(X, y)


# Borutaアルゴリズムによる特徴量選択
boruta_selector = BorutaPy(
    model,
    n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
    verbose=2,
    alpha=0.05,  # 有意水準
    max_iter=100,  # 試行回数
    perc=90,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
    two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
    random_state=0
)

boruta_selector.fit(np.array(X), np.array(y))

# 有効な特徴量の選択
selected_features = X.columns[boruta_selector.support_].tolist()
X_selected = X[selected_features]


# モデルの訓練と評価のための関数
def train_and_evaluate_model(X, y, model):
    skf_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in skf_outer.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        tprs_inner = []
        aucs_inner = []
        for train_inner_index, test_inner_index in skf_inner.split(X_train, y_train):
            X_train_inner, X_test_inner = X_train.iloc[train_inner_index], X_train.iloc[test_inner_index]
            y_train_inner, y_test_inner = y_train.iloc[train_inner_index], y_train.iloc[test_inner_index]

            model.fit(X_train_inner, y_train_inner)
            y_pred_inner = model.predict_proba(X_test_inner)[:, 1]

            fpr, tpr, _ = roc_curve(y_test_inner, y_pred_inner)
            tprs_inner.append(np.interp(mean_fpr, fpr, tpr))
            tprs_inner[-1][0] = 0.0
            roc_auc_inner = auc(fpr, tpr)
            aucs_inner.append(roc_auc_inner)

        mean_tpr_inner = np.mean(tprs_inner, axis=0)
        mean_tpr_inner[-1] = 1.0
        mean_auc_inner = np.mean(aucs_inner)

        tprs.append(mean_tpr_inner)
        aucs.append(mean_auc_inner)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    print("mean_auc: ", mean_auc)

    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tpr + std_auc, 1)
    tprs_lower = mean_tpr - std_auc
    print("standard deviation: ", std_auc)

    # ROC曲線の描画
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.3, label='Standard deviation')

    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')


# モデルのリストを作成
models = []
for _ in range(400):
    model = xgb.XGBClassifier(**best_params)
    models.append(model)

# 400のユニークなモデルに対してrNCVを実行
for model in models:
    train_and_evaluate_model(X_selected, y, model)

plt.show()

explainers = []
shap_values_list = []

for i in range(100):
    # XGBoostモデルの訓練
    model.fit(X_selected, y)

    # SHAP値の計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_selected)

    explainers.append(explainer)
    shap_values_list.append(shap_values)

# SHAP値の平均化
avg_shap_values = np.mean(shap_values_list, axis=0)
# SHAPプロットの作成
shap.summary_plot(avg_shap_values, X_selected)
