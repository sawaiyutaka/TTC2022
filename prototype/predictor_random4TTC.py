import pandas as pd
import numpy as np
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

TRIAL = 1000  # ランダム化を何回するか
FEATURE = "AA205_Imp"  # 調べたい項目
warnings.filterwarnings('ignore')
"""
# 現在の最大表示列数の出力
pd.get_option("display.max_columns")

# 最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_columns', 500)
"""
df = pd.read_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_alldata_CATE_4th.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df.head(10))

shap.initjs()  # いくつかの可視化で必要

# 特徴量 X、アウトカム y
y = df["te_pred"]
X = df.drop(["te_pred", "Unnamed: 0"], axis=1)
print(X.head())
print(y)

Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習

j = X_val.columns.get_loc(FEATURE)  # カラム数を抽出
print("FEATUREの列: ", j)

ls0 = []
for i in range(TRIAL):
    model = RandomForestRegressor(random_state=i)

    model.fit(X_train, Y_train)

    # 学習済みモデルの評価
    predicted_Y_val = model.predict(X_val)
    print("ランダム化前のmodel_score: ", model.score(X_val, Y_val))

    # shap valueで評価

    # 短時間の近似式
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:100])

    # print(shap_values)

    # print("shap value\n", shap_values)

    ls0.append(np.abs(shap_values[:, j]).mean())

    print("random_state ", i, ": ランダム化前のshap value近似値\n", np.abs(shap_values[:, j]).mean())


true_shap = pd.DataFrame(ls0, columns=['shap_value'])
true_shap['color'] = 1

"""
# Fits the explainer
explainer = shap.Explainer(model.predict, X_val)

# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val[:100], max_evals=2000)
true_shap = shap_values[:, FEATURE].abs.mean(0).values
print("ランダム化前のshap value\n", true_shap)

"""


# yをランダム化
ls = []
for i in range(TRIAL):
    y = y.sample(frac=1, random_state=i)
    # print("ランダム化", i+1, "回目のy:\n", y)

    Y_train, Y_val, X_train, X_val = train_test_split(y, X, test_size=.2)

    # sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
    # 教師データと教師ラベルを使い、fitメソッドでモデルを学習

    model.fit(X_train, Y_train)

    # 学習済みモデルの評価
    predicted_Y_val = model.predict(X_val)
    print("ランダム化後のmodel_score: ", model.score(X_val, Y_val))

    # shap valueで評価
    """
    # Fits the explainer
    explainer = shap.Explainer(model.predict, X_val)
    # Calculates the SHAP values - It takes some time
    shap_values = explainer(X_val[:100], max_evals=2000)
    print(i + 1, "回目のランダム化のshap value\n", shap_values[:, FEATURE].abs.mean(0).values)

    ls.append(shap_values[:, FEATURE].abs.mean(0).values)
    """

    # treeexplainerで近似（早い）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:100])  # , check_additivity=False)  # 数が少ないとSHAPの予測が不正確になるためエラーになる

    # TreeEcplainerのとき
    print(i + 1, "回目のランダム化のshap value近似値\n", np.abs(shap_values[:, j]).mean())
    ls.append(np.abs(shap_values[:, j]).mean())

print(ls)
# 数値的に上下5%の値をみてみる
print("95%CI of shap at random: ", np.quantile(ls, [0.025, 0.975]))

# まず1000回分をプロット
# s.set()
# s.displot(ls)
# plt.show()

# true shapと合わせてプロット
shap_random = pd.DataFrame(ls, columns=['shap_value'])
shap_random['color'] = 0

df4plot = pd.concat([shap_random, true_shap])
df4plot["shap_value"] = np.sqrt(df4plot["shap_value"])

s.displot(data=df4plot, x='shap_value', hue='color', multiple='stack')
plt.savefig("/Volumes/Pegasus32R8/TTC/202211/trial_1000vs1000.svg")
plt.show()
