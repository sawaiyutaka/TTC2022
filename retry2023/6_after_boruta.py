from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2023retry/ocs2ple.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

y = df["group"]
print(y)

X_selected = df[["AA165", "AA123", "AD27_7", "AE1BMI", "AB62_7", "AA84", "AB149", "AC25", "AB116", "AA86",
                 "AA101", "BR23UMU", "AB64", "AB70"]]  # borutaで選択した変数を
print("Xの項目数", X_selected.shape)
print(X_selected.head())


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
    'n_estimators': list(range(101)),
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [1, 2],
    'random_state': [42],
    'n_jobs': [int(cpu_count() / 2)]
}
# best_score:  0.28219696969696967
# best_params:  {'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 92, 'n_jobs': 18, 'random_state': 42}

# Youden Indexを最大化するためのスコア関数を作成
scoring = make_scorer(youden_index_score, greater_is_better=True)

skf_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
# rNCVでのハイパーパラメータの探索
best_score = 0
best_params = {}

# 100回繰り返す
repeats = 100
roc_curves = []

for _ in range(repeats):
    tprs_outer = []
    aucs_outer = []
    # 各モデルのROC曲線を計算し、結果を保持するリスト

    for train_index, test_index in skf_outer.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

        # 最も性能の良いモデルを保存
        if score > best_score:
            best_score = score
            best_params = params

        # 最終的な分類器の訓練
        model = xgb.XGBClassifier(**params)
        model.fit(X_selected, y)

        y_pred = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        # ROC曲線をリストに追加
        roc_curves.append((fpr, tpr))

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    tprs.extend(tprs_outer)
    aucs.extend(aucs_outer)

print("best_score: ", best_score)
print("best_params: ", best_params)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
print("mean_auc: ", mean_auc)

std_auc = np.std(aucs)
print("standard deviation: ", std_auc)
# mean_auc:  0.8665940611664297
# standard deviation:  0.029762172277986467

# FPR の一意な値を取得
unique_fpr = np.unique(np.concatenate([fpr for fpr, _ in roc_curves]))
mean_tpr = np.mean([np.interp(unique_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)
std_tpr = np.std([np.interp(unique_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)

# ROC曲線の描画
plt.plot(unique_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
plt.fill_between(unique_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='gray', alpha=0.2,
                 label='Standard Deviation')

plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

explainers = []
shap_values_list = []
model = xgb.XGBClassifier(**best_params)

for i in range(100):
    # サンプル全体に対して、最終モデルのXGBoostモデルで訓練
    model.fit(X_selected, y)

    # SHAP値の計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_selected)

    explainers.append(explainer)
    shap_values_list.append(shap_values)

# SHAP値の平均化
avg_shap_values = np.mean(shap_values_list, axis=0)
# SHAPプロットの作成
shap.summary_plot(avg_shap_values, X_selected, max_display=len(X_selected.columns))
