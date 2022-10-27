import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as s

# モデルはRandom Forestを使う
from sklearn.ensemble import RandomForestRegressor

# SHAP(SHapley Additive exPlanations)
import shap
# shap 0.41だとエラー（shap_valuesが廃止されている）

from sklearn.model_selection import train_test_split, GridSearchCV

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("//Volumes/Pegasus32R8/TTC/2022csv/TTC2022_CATE_shap.csv", delimiter=",")
print(df.head(10))

shap.initjs()  # いくつかの可視化で必要

# 特徴量 X、アウトカム y
y = df["te_pred"]
X = df.drop(["te_pred"], axis=1)
print(X.head())
print(y)

Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model = RandomForestRegressor(
    max_depth=30,
    max_features=25,
    min_samples_split=5,
    n_estimators=500,
    n_jobs=1,
    random_state=2525)

model.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model.predict(X_val)
print("model_score: ", model.score(X_val, Y_val))

# shap valueで評価
# Fits the explainer
explainer = shap.Explainer(model.predict, X_val)

# 短時間の近似式
# explainer = shap.TreeExplainer(model)

# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val[:100])

# print(shap_values)

str = "AB61"  # 調べたい項目
true_shap = shap_values[:, str].abs.mean(0).values
print("ランダム化前のshap value\n", true_shap)


# yをランダム化
ls = []
for i in range(1000):
    y = y.sample(frac=1, random_state=i)
    # print("ランダム化", i+1, "回目のy:\n", y)

    Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

    # sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
    # 教師データと教師ラベルを使い、fitメソッドでモデルを学習
    model = RandomForestRegressor(
        max_depth=30,
        max_features=25,
        min_samples_split=5,
        n_estimators=500,
        n_jobs=1,
        random_state=2525)
    model.fit(X_train, Y_train)

    # 学習済みモデルの評価
    # predicted_Y_val = model.predict(X_val)
    # print("model_score: ", model.score(X_val, Y_val))

    # shap valueで評価
    # Fits the explainer
    explainer = shap.Explainer(model.predict, X_val)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_val[:100])

    print(i + 1, "回目のランダム化のshap value\n", shap_values[:, str].abs.mean(0).values)

    ls.append(shap_values[:, str].abs.mean(0).values)

    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_val[:100])  # , check_additivity=False)  # 数が少ないとSHAPの予測が不正確になるためエラーになる

    # TreeEcplainerのとき
    # print(i+1, "回目のランダム化のshap value\n", np.abs(shap_values[:, j]).mean())
    # ls.append(np.abs(shap_values[:, j]).mean())

print(ls)
# 数値的に上下5%の値をみてみる
print("95%CI of shap at random: ", np.quantile(ls, [0.05, 0.95]))

# まず1000回分をプロット
s.set()
s.displot(ls)
plt.show()

# true shapと合わせてプロット
shap_random = pd.DataFrame(ls, columns=['shap_value'])
shap_random['color'] = 0

shap_true = pd.DataFrame([[true_shap, 1]], columns=['shap_value', 'color'])
df4plot = pd.concat([shap_random, shap_true])

s.displot(data=df4plot, x='shap_value', hue='color', multiple='stack')
plt.show()


