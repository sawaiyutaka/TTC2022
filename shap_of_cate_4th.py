import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib

# モデルはRandom Forestを使う
from sklearn.ensemble import RandomForestRegressor

# SHAP(SHapley Additive exPlanations)
import shap

shap.initjs()  # いくつかの可視化で必要
from sklearn.model_selection import train_test_split, GridSearchCV

# with codecs.open("gender_gap_full.csv", "r", "Shift-JIS", "ignore") as file:
#    df = pd.read_table(file, delimiter=",")

# 特徴量 X、アウトカム y
df = pd.read_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_alldata_CATE_4th.csv", delimiter=",")
print(df.head())
y = df["te_pred"]
print(y)

X = pd.read_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/X_imputed.csv", delimiter=",")
X = X.drop(["Unnamed: 0", "SAMPLENUMBER"], axis=1)
print(X.head())

Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model = RandomForestRegressor(
    max_depth=None,
    max_features=200,  # X_train.shape[1],  # The number of features to consider when looking for the best split
    # 'sqrt'も可能
    min_samples_split=2,
    min_samples_leaf=1,
    n_estimators=2000,
    # n_jobs=8,  # number of jobs to run in parallel(-1 means using all processors)
    random_state=42)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

model.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model.predict(X_val)
print("model_score: ", model.score(X_val, Y_val))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(model.predict, X_val)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val, max_evals=2000)  # max_evals低すぎるとエラー

shap.plots.bar(shap_values, max_display=14)
shap.plots.beeswarm(shap_values, max_display=14)


"""
# ハイパーパラメータをチューニング
search_params = {
    'n_estimators': [500, 1000, 2000],
    'max_features': [i for i in range(100, X_train.shape[1], 50)],
    'random_state': [2525],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20]
    # 'max_depth': [20, 30, 40]
    # Best model score 0.57(max_features=200, n_estimators=2000, random_state=2525)
}

# グリッドサーチ
gsr = GridSearchCV(
    RandomForestRegressor(),
    search_params,
    n_jobs=16,  # max18core
    cv=3,  # Determines the cross-validation splitting strategy
    verbose=True  # Controls the verbosity when fitting and predicting.
)

gsr.fit(X_train, Y_train)

# 最もよかったモデル
print(gsr.best_estimator_)
print("最もよかったモデルの評価", gsr.best_estimator_.score(X_val, Y_val))
"""