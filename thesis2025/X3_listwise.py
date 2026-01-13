import glob
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# =========================
# 1) 読込
# =========================
df0 = pd.read_table(
    r"D:\documents\UT\thesis\before_impute.csv",
    delimiter=",",
    low_memory=False
).set_index("SAMPLENUMBER")

print(df0.shape)
print(df0.head())

# =========================
# 2) 目的変数（PE_14_16）を作成して、0/1確定者のみ抽出
# =========================
# ここは「ple_cols」をあなたのデータに合わせて必ず定義してください
ple_cols = ["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
            "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1"]

# 必要列があるかチェック
missing_cols = [c for c in ple_cols if c not in df0.columns]
if missing_cols:
    raise ValueError(f"PLE列が見つかりません: {missing_cols}")

ple_condition = df0[ple_cols].gt(2).any(axis=1)   # どれか1つでも >2
non_condition = df0[ple_cols].lt(2).all(axis=1)   # 全部 <2

df0["PE_14_16"] = np.select([ple_condition, non_condition], [1, 0], default=np.nan)

# PEが0/1に確定した人だけ
df = df0[df0["PE_14_16"].isin([0, 1])].copy()

# OCS_0or1==1 の人だけ
if "OCS_0or1" not in df.columns:
    raise ValueError("OCS_0or1 列が見つかりません。列名を確認してください。")
df = df[df["OCS_0or1"] == 1].copy()

# 目的変数
y = df["PE_14_16"].astype(int)
print(y.value_counts(dropna=False))

# bulliedが無い場合のみ、あなたの定義（AB61/AD19）で作る
if "bullied" not in df.columns:
    if all(c in df.columns for c in ["AB61", "AD19"]):
        mask = df[["AB61", "AD19"]].notna().all(axis=1)
        df["bullied"] = np.nan
        df.loc[mask, "bullied"] = (df.loc[mask, ["AB61", "AD19"]].lt(5).any(axis=1)).astype(int)
    else:
        print("注意: bullied列が無く、作成に必要なAB61/AD19も見つかりませんでした。bulliedは欠損扱いになります。")
        df["bullied"] = np.nan

# =========================
# 3) 説明変数（12変数）
# =========================
X_selected = df[[
    "bullied", "AD27_7", "AA97", "AD3", "AB46", "AB250",
    "AB64", "AB54", "AA86", "AB12.5", "AB72", "AB186Ln(TD)"
]].copy()

# 値の向きをそろえる（列ごとに明示的に置換）
# 「AD27_7」は 1/0 が逆転（コメント通り）にしたい場合
X_selected["AD27_7"] = X_selected["AD27_7"].replace({1: 0, 0: 1})

# 反転スケール
X_selected["AD3"]   = X_selected["AD3"].replace({1: 4, 2: 3, 3: 2, 4: 1})
X_selected["AB46"]  = X_selected["AB46"].replace({1: 4, 2: 3, 3: 2, 4: 1})
X_selected["AB250"] = X_selected["AB250"].replace({1: 2, 2: 1})
X_selected["AB64"]  = X_selected["AB64"].replace({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})

print("Xの項目数:", X_selected.shape)
print(X_selected.head())

# =========================
# 4) Youden index（GridSearch用：予測ラベルが必要なので0.5閾値で2値化）
# =========================
def youden_index_score(y_true, y_proba, threshold=0.5):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return sensitivity + specificity - 1

# GridSearchCV は (y_true, y_pred) 形式で scorer が呼ばれるため、
# predict_proba の[:,1] を渡す想定の scorer をラップ
def youden_scorer(estimator, X, y_true):
    y_proba = estimator.predict_proba(X)[:, 1]
    return youden_index_score(y_true, y_proba, threshold=0.5)

# =========================
# 5) rNCV（4-fold×4-fold を 100回）
# =========================
param_grid = {
    "n_estimators": [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "learning_rate": [0.1, 0.2, 0.3],
    "max_depth": [1, 2],
    "random_state": [42],
    "n_jobs": [max(1, int(cpu_count() / 2))]
}

repeats = 100
roc_curves = []
aucs = []

best_score = -np.inf
best_params = None

# SHAPを「外側テスト部分」ごとに貯める（リーク回避のため全データで計算しない）
shap_values_list = []
X_for_shap_list = []

for i in range(repeats):
    print(f"{i + 1} out of {repeats}")

    skf_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + i)

    for train_index, test_index in skf_outer.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1000 + i)

        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(
                eval_metric="logloss",
            ),
            param_grid=param_grid,
            scoring=youden_scorer,
            cv=inner_cv
        )
        grid_search.fit(X_train, y_train)

        params = grid_search.best_params_
        score = grid_search.best_score_

        if score > best_score:
            best_score = score
            best_params = params

        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        roc_curves.append((fpr, tpr))
        aucs.append(roc_auc)

        # SHAP（外側テストだけ）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_values_list.append(shap_values)
        X_for_shap_list.append(X_test)

print("best_score (Youden on inner CV):", best_score)
print("best_params:", best_params)

mean_auc = float(np.mean(aucs))
std_auc = float(np.std(aucs))
print("mean_auc:", mean_auc)
print("standard deviation:", std_auc)

# =========================
# 6) ROC曲線（平均±SD）
# =========================
unique_fpr = np.unique(np.concatenate([fpr for fpr, _ in roc_curves]))
interp_tprs = np.array([np.interp(unique_fpr, fpr, tpr) for fpr, tpr in roc_curves])

mean_tpr = interp_tprs.mean(axis=0)
std_tpr = interp_tprs.std(axis=0)

plt.plot(unique_fpr, mean_tpr, label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
plt.fill_between(unique_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, label="SD")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# =========================
# 7) SHAP summary（外側テストのSHAPを結合して表示）
# =========================
X_for_shap = pd.concat(X_for_shap_list, axis=0)
shap_values_all = np.vstack(shap_values_list)

shap.summary_plot(shap_values_all, X_for_shap, max_display=len(X_selected.columns))
