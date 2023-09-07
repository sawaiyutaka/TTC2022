import glob
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2023retry/ocs2ple_w_imp.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# プログラム2｜所定フォルダ内の「data*.xlsx」を取得
files = glob.glob('/Volumes/Pegasus32R8/TTC/2022base_OC_PLE/db*.xlsx')

# プログラム3｜変数listを空リストで設定
ls = []

# プログラム4｜プログラム2で取得したエクセルを一つずつpandasデータとして取得
for file in files:
    d = pd.read_excel(file)
    print(file)
    d = d.set_index("SAMPLENUMBER")
    # print(d)
    ls.append(d)

# プログラム5｜listに格納されたエクセルファイルをpandasとして結合
oc_4th = pd.concat(ls, axis=1, join='inner')
print(oc_4th)
# 第4期のOCS
# oc_4th = oc_4th[["DB57", "DB100"]]  # 第4期強迫観念、強迫行為
df_ocs4th = df.join(oc_4th, how='inner')
print(df_ocs4th)
df_ocs4th.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/ocs2ple_w_ocs4th.csv")

y = df["group"]
print(y)

X_selected = df[[
    "bullied", "AD27_7", "AA97", "AD3", "AB46", "AB250", "AB64", "AB54", "AA86", "AB12.5", "AB72", "AB186Ln(TD)"
]]
# 時間割引率はLnの値が小さいほど、割引率小さい（がまんできる）
X_selected = X_selected.replace({"AD27_7": {1: 0, 0: 1}})
X_selected = X_selected.replace({"AD3": {1: 4, 2: 3, 3: 2, 4: 1}})
X_selected = X_selected.replace({"AB46": {1: 4, 2: 3, 3: 2, 4: 1}})
X_selected = X_selected.replace({"AB250": {1: 2, 2: 1}})
X_selected = X_selected.replace({"AB64": {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}})
X_selected = X_selected.replace({"AB46": {1: 4, 2: 3, 3: 2, 4: 1}})

# ocs2ple brutaで150回以上抽出
# "AD27_7", "AA165", "AA101", "AA97", "AA84", "bullied", "AB12.5", "A213SSQS_Imp", "AB116", "AB46", "AB54",
# "AB149", "AB64", "AE1BMI", "AB158"
# ple borutaで100/400回以上抽出された変数を使う
# "AA101", "AB126", "bullied", "AB127", "AA94", "AE13.8", "BR46UMU", "AB86", "AA219", "AB149", "AD27_4", "AA114",
# "AEIQ", "AC42", "AD62", "AB114", "AB233.1", "AC66", "AB65_4"
# ocsの予測で
# "AB71", "AB102", "AA90", "AB154", "AB148", "AB96", "AQ_sum", "AB149", "AA100", "AB87",
# "AA106", "AC45_1", "AC25", "AB147", "AD75", "AA85", "AB72", "AB128", "AB89", "AB143", "Webaddict4", "AA86"
# ocs2ple pleはimputationなしで、101回以上borutaで抽出された項目
# "bullied", "AD27_7", "AA97", "AD3", "AB46", "AB250", "AB64", "AB54", "AA86", "AB12.5", "AB72", "AB186Ln(TD)"

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
    'n_estimators': [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # list(range(101)),
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [1, 2],
    'random_state': [42],
    'n_jobs': [int(cpu_count() / 2)]
}

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
explainers = []
shap_values_list = []
for i in range(repeats):
    print(i + 1, "out of 100")
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
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        # ROC曲線をリストに追加
        roc_curves.append((fpr, tpr))

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_selected)

        explainers.append(explainer)
        shap_values_list.append(shap_values)

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
# best_score:  0.6222943722943723
# best_params:  {'learning_rate': 0.2, 'max_depth': 2, 'n_estimators': 100, 'n_jobs': 18, 'random_state': 42}
# mean_auc:  0.8880955725462303
# standard deviation:  0.04290517810557535

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

# SHAP値の平均化
avg_shap_values = np.mean(shap_values_list, axis=0)
# SHAPプロットの作成
shap.summary_plot(avg_shap_values, X_selected, max_display=len(X_selected.columns))
