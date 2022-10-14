import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt

# モデルはRandom Forestを使う
from sklearn.ensemble import RandomForestRegressor

# SHAP(SHapley Additive exPlanations)
import shap

from sklearn.model_selection import train_test_split, GridSearchCV

# with codecs.open("gender_gap_full.csv", "r", "Shift-JIS", "ignore") as file:
#    df = pd.read_table(file, delimiter=",")

df = pd.read_csv("/Volumes/Pegasus32 R8/TTC/2022csv", delimiter=",")
print(df.head(10))

shap.initjs()  # いくつかの可視化で必要

print(df.head(15))

# 特徴量 X、アウトカム y
y = df["te_pred"]
X = df.drop(["te_pred"], axis=1)
print(X.head())
print(y)

Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model = RandomForestRegressor(
    max_depth=20,
    max_features=24,
    min_samples_split=20,
    n_estimators=300,
    n_jobs=-1,
    random_state=2525)

model.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model.predict(X_val)
print("model_score: ", model.score(X_val, Y_val))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(model.predict, X_val)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val)

shap.plots.bar(shap_values, max_display=20)
shap.plots.beeswarm(shap_values, max_display=20)
# ハイパーパラメータをチューニング
search_params = {
    'n_estimators': [300, 500, 1000, 10000],
    'max_features': [i for i in range(1, X_train.shape[1])],
    'random_state': [2525],
    'n_jobs': [1, -1],
    'min_samples_split': [5, 10, 20],
    'max_depth': [10, 20, 30]
}

# グリッドサーチ
gsr = GridSearchCV(
    RandomForestRegressor(),
    search_params,
    cv=3,
    verbose=True
)

gsr.fit(X_train, Y_train)

# 最もよかったモデル
print(gsr.best_estimator_)
print("最もよかったモデルの評価", gsr.best_estimator_.score(X_val, Y_val))

"""
print("非労働人口のshap: ", shap_values[:, "非労働力人口【人】"].abs.mean(0).values)

# TreeExplainerで計算（やや早い）
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val[:100])  # , check_additivity=False)  # 数が少ないとSHAPの予測が不正確になるためエラーになる
"""

"""
print("shap value\n", shap_values)
j = X.columns.get_loc("非労働力人口【人】")  # カラム数を抽出
print("〇〇の列名: ", j)
print("〇〇のshap value絶対値の平均\n", np.abs(shap_values[:, j]).mean())
"""