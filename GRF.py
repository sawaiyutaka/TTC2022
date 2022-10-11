import pandas as pd
import numpy as np
import seaborn as s
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLassoCV, LassoCV
from sklearn.model_selection import train_test_split, GridSearchCV

from missingpy import MissForest

from econml.dml import CausalForestDML
import shap

# df = pd.read_table("TTC2022_ple_naive.csv", delimiter=",")
# df = df.drop(["Unnamed: 0"], axis=1)
# print(df.head())

"""
# Make an instance and perform the imputation
imputer = MissForest()
df_imputed = imputer.fit_transform(df)
print("df_imputed\n", df_imputed)
df[df.columns.values] = df_imputed
df = df.round().astype(int)  # 各列を整数に丸める（身長、体重も丸め）
# df.to_csv("TTC2022_ple_naive_imputed.csv")
print(df)
"""

"""
# imputeした後のデータフレーム、PLEとAQの合計得点前
df = pd.read_table("TTC2022_ple_naive_imputed.csv", delimiter=",")
print(df.head())

# PLEの合計点を作成
df_Y = df[["SAMPLENUMBER", "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"]]
print("df_Y\n", df_Y)
# df_Y = df_Y.fillna(1) # NaNは「なかった」とみなす
df_Y = df_Y.set_index("SAMPLENUMBER")
df_Y["PLE_sum"] = df_Y.sum(axis=1)
print("第3回PLE合計\n", df_Y["PLE_sum"])
df_Y = df_Y.reset_index()

# AQの合計点を作成
df_AQ = df.filter(regex='^(SAMPLENUMBER|BB12|BB13)', axis=1)
df_AQ = df_AQ.set_index("SAMPLENUMBER")
df_AQ["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df_AQ["AQ_sum"])
df_AQ = df_AQ.reset_index()

df = pd.merge(df, df_Y[["SAMPLENUMBER", "PLE_sum"]], on="SAMPLENUMBER")
print("PLE合計点を追加した\n", df.head(15))

df = pd.merge(df, df_AQ[["SAMPLENUMBER", "AQ_sum"]], on="SAMPLENUMBER")
print("AQ合計点を追加した\n", df.head(23))
# df.to_csv("TTC2022_PLE_sum.csv")
"""

# impute後、AQとPLE合計点を計算後
df = pd.read_table("TTC2022_PLE_sum.csv", delimiter=",")

# 特徴量 X、アウトカム y、割り当て変数 T
Y = df['PLE_sum']

# 他のPLE項目でどうなるか
# y = df['CD65_1']

print("Y\n", df["PLE_sum"].describe())
T = df['OCS_0or1']  # 強迫5点以上をtreatmentとする

'''
X_NaN = df.drop(["PLE_sum", "OCS_0or1", "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"], axis=1)

# Make an instance and perform the imputation
imputer = MissForest()
X_imputed = imputer.fit_transform(X_NaN)
print("X_imputed\n", X_imputed)
X_NaN[X_NaN.columns.values] = X_imputed
X_NaN.to_csv("TTC2022_X_imputed_ple_naive.csv")

# X_NaN = pd.read_table("TTC2022_X_imputed_ple_naive.csv", delimiter=",")


# 第1期の強迫、PLEを除外
X = df.filter(regex='^(A)', axis=1)
print(X)
X = X.drop(["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"], axis=1)
X = X.drop(["AD57", "AD58", "AD59", "AD60", "AD61", "AD62"], axis=1)
X.to_csv("TTC2022_X_dummy.csv")
# X = df.drop(["SAMPLENUMBER", "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1", 'PLE_sum', "OCS_0or1", "OCS_sum", axis=1)

# AB基本セットのみ使用する場合
# X_col_use = pd.read_table("TTC2022_base_minimum.csv", delimiter=",")
# X = df[X_col_use.columns.values]  
'''

# 第1期の強迫、PLEを除外したXを読み込み
X = pd.read_table("TTC2022_X_dummy.csv", delimiter=",")
print("X:\n", X.head(10))

# print("補完後のNaN個数\n", PLE_imputed.isnull().sum())

# https://github.com/microsoft/EconML/blob/main/notebooks/Generalized%20Random%20Forests.ipynb
# 1. Causal Forest: Heterogeneous causal effects with no unobserved confounders


n_samples = len(df)
n_treatments = 1

Y_train, Y_val, T_train, T_val, X_train, X_val = train_test_split(Y, T, X, test_size=.2)
W = None

"""
est = CausalForestDML(model_y=RandomForestRegressor(),
                      model_t=RandomForestRegressor(),
                      criterion='mse',
                      n_estimators=1000,
                      max_depth=40,
                      min_samples_split=20,
                      min_impurity_decrease=0.001,
                      random_state=42)
"""

causal_forest = CausalForestDML(criterion='het',
                                n_estimators=10000,
                                min_samples_leaf=10,
                                max_depth=None,
                                max_samples=0.5,
                                discrete_treatment=False,
                                honest=True,
                                inference=True,
                                cv=10,
                                model_t=LassoCV(),
                                model_y=LassoCV(),
                                )

# fit train data to causal forest model
causal_forest.fit(Y, T, X=X, W=W)

# estimate the CATE with the test set
causal_forest.const_marginal_ate(X_train)

"""
# SHAP
shap_values = est.shap_values(X)
shap.plots.beeswarm(shap_values['y']['T'])
plt.show()
"""

# 半分に分割してテスト
# test1
X_test1 = X.iloc[:int(n_samples / 2), :]
# test2
X_test2 = X.iloc[int(n_samples / 2):n_samples, :]

# X_test[:, 0] = np.linspace(np.percentile(X[:, 0], 1), np.percentile(X[:, 0], 99), min(100, n_samples))
print("X_test1: \n", X_test1)
print("X_test2: \n", X_test2)

# 半分に分割時
# print("X_test: \n", X_test)

# X全体でCATEを計算
te_pred = causal_forest.effect(X)

# X_testのみでCATEを計算
te_pred_test1 = causal_forest.effect(X_test1)
te_pred_test2 = causal_forest.effect(X_test2)

print("te_pred: \n", te_pred)
print("要素数", len(te_pred))
# 各CATEの値のXの要素を示す
df_new = df.assign(te_pred=te_pred)
print("CATEを追加\n", df_new)
# df_new.to_csv("TTC2022_CATE.csv")

# CATEの推定結果を確認
print("CATE of CausalForest: ", round(np.mean(te_pred), 2))

print("Percentile of CATE of CausalForest: 10%, 25%, 50%, 75%, 90%\n",
      np.quantile(a=te_pred, q=[.1, .25, .5, .75, .9]))

upper = np.quantile(a=te_pred, q=.9)  # CATE上位10％の境目
lower = np.quantile(a=te_pred, q=.1)  # CATE下位10％の境目
df_upper = df_new[(df_new["te_pred"] > upper)]  # CATE上位10%
df_lower = df_new[(df_new["te_pred"] < lower)]  # CATE下位10%
print("upeer＝影響を受けやすかった10%: \n", df_upper)
print("lower＝影響を受けにくかった10%: \n", df_lower)

print("df_upper\n", df_upper.describe())
# df_upper.to_csv("TTC2022_upper.csv")

print("df_lower\n", df_lower.describe())
# df_lower.to_csv("TTC2022_lower.csv")

# CATE(全体)
s.set()
s.displot(te_pred)
plt.show()


# CATE(前半)
s.displot(te_pred_test1)
plt.show()

# CATE(後半)
s.displot(te_pred_test2)
plt.show()


'''
# https://towardsdatascience.com/causal-machine-learning-for-econometrics-causal-forests-5ab3aec825a7
# ★['Y0']['T0']問題！
# fit causal forest with default parameters
causal_forest = CausalForestDML()
causal_forest.fit(Y, T, X=X, W=W)

# calculate shap values of causal forest model
shap_values = causal_forest.shap_values(X)
# plot shap values
# shap.summary_plot(shap_values['Y0']['T0'])
# shap.summary_plot(shap_values, X)ではダメ
'''

# Note that the structure of this estimator is based on the BaseEstimator and RegressorMixin from sklearn; however,
# here we predict treatment effects –which are unobservable– hence regular model validation and model selection
# techniques (e.g. cross validation grid search) do not work as we can never estimate a loss on a training sample,
# thus a tighter integration into the sklearn workflow is unlikely for now.
