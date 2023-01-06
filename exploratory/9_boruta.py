import sys
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
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

df = pd.read_table("df_3rd.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# hyperparameter tuning using Optuna with RandomForestClassifier Example (Python code)
# https://www.datasciencebyexample.com/2022/07/02/2022-07-02-1/

# 特徴量 X、アウトカム y、割り当て変数 T
y = df["PLE_sum_3rd"]
X = df.drop(["PLE_sum_3rd"], axis=1)
print(X.head())
print(y)
Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=.2)

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model = RandomForestRegressor(max_depth=None,
                              max_features='sqrt',
                              # X_train.shape[1],  # The number of features to consider when looking for the best split
                              # 'sqrt'も可能
                              min_samples_split=5,
                              min_samples_leaf=1,
                              n_estimators=5000,
                              n_jobs=int(cpu_count() / 2),
                              random_state=42)
model.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model.predict(X_test)
print("model_score: ", model.score(X_test, Y_test))

# Borutaを実行
rf = RandomForestRegressor(n_jobs=int(cpu_count() / 2), max_depth=7)
feat_selector = BorutaPy(rf, n_estimators='auto', two_step=False, verbose=2, random_state=42)
feat_selector.fit(X_train.values, Y_train.values)
print(X_train.columns[feat_selector.support_])

# 選択したFeatureを取り出し
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_val_selected = X_test.iloc[:, feat_selector.support_]
print(X_val_selected.head())

# 選択したFeatureで学習
rf2 = RandomForestRegressor(max_depth=None,
                            max_features='auto',
                            # X_train.shape[1],  # The number of features to consider when looking for the best split
                            # 'sqrt'も可能
                            min_samples_split=5,
                            min_samples_leaf=1,
                            n_estimators=5000,
                            n_jobs=int(cpu_count() / 2),
                            random_state=42)
rf2.fit(X_train_selected.values, Y_train.values)

predicted_Y_val_selected = rf2.predict(X_val_selected.values)
print("model_score_2: ", rf2.score(X_val_selected, Y_test))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(rf2.predict, X_val_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val_selected)

shap.plots.bar(shap_values, max_display=len(X_val_selected.columns))
shap.summary_plot(shap_values, max_display=len(X_val_selected.columns))
