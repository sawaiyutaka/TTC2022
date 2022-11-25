import pandas as pd
import numpy as np
import seaborn as s
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
import shap

# imputeした後のデータフレーム、PLEとAQの合計得点前
df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/base_ple_imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

"""
# PLEの合計点を作成(第3期)
df_Y = df[["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"]]
print("df_Y\n", df_Y)
df_Y["PLE_sum"] = df_Y.sum(axis=1)
print("第3回PLE合計\n", df_Y["PLE_sum"])
# df_Y = df_Y.reset_index()
"""

# PLEの合計点を作成(第4期)
df_Y = df[["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"]]
print("df_Y\n", df_Y)
df_Y["PLE_sum"] = df_Y.sum(axis=1)
print("第4回PLE合計\n", df_Y["PLE_sum"])
# df_Y = df_Y.reset_index()

# AQの合計点を作成
df_AQ = df.filter(regex='^(BB12|BB13)', axis=1)
df_AQ["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df_AQ["AQ_sum"])
# df_AQ = df_AQ.reset_index()

df = pd.concat([df, df_Y], axis=1, join='inner')
print("PLE合計点を追加した\n", df.head())

df = pd.concat([df, df_AQ], axis=1, join='inner')
print("AQ合計点を追加した\n", df.head())
# df.to_csv("TTC2022_PLE_sum.csv")


# 特徴量 X、アウトカム Y、割り当て変数 T
Y = df_Y['PLE_sum']  # 'CD65_1'などとすると、単一項目で見られる
print("Y\n", Y)

T = df['OCS_0or1']  # 強迫CMCL5点以上であることをtreatmentとする

# 第3期のPLEを除外
X = df.drop(["PLE_sum", "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"], axis=1)

# 第4期のPLEを除外
X = X.drop(["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)

# 第２期の強迫を除外
X = X.drop(["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116", "OCS_sum", "OCS_0or1"], axis=1)

# 第２期のAQを除外
X = X.drop(["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"], axis=1)

print(X)
# 第1期の強迫を除外
X = X.drop(["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"], axis=1)
X = X.drop(["AD57", "AD58", "AD59", "AD60", "AD61", "AD62"], axis=1)

X = X.loc[:, ~X.columns.duplicated()]
print("重複を削除\n", X)
X.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/X_imputed.csv")


# 第1期の強迫、PLEを除外したXを読み込み
# X = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_alldata/X_imputed.csv", delimiter=",")
print("X:\n", X)
print("補完後のNaN個数\n", X.isnull().sum())

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
                                model_t=LassoCV(max_iter=100000),
                                model_y=LassoCV(max_iter=100000),
                                random_state=2525,
                                n_jobs=10)

# fit train data to causal forest model
causal_forest.fit(Y, T, X=X, W=W)

# estimate the CATE with the test set
causal_forest.const_marginal_ate(X_train)

'''
# 半分に分割してテスト
# test1
X_test1 = X.iloc[:int(n_samples / 2), :]
# test2
X_test2 = X.iloc[int(n_samples / 2):n_samples, :]

print("X_test1: \n", X_test1)
print("X_test2: \n", X_test2)
'''
# X全体でCATEを計算
te_pred = causal_forest.effect(X)
# lb, ub = causal_forest.effect_interval(X, alpha=0.05)

'''
# X_testのみでCATEを計算
te_pred_test1 = causal_forest.effect(X_test1)
te_pred_test2 = causal_forest.effect(X_test2)
'''

print("te_pred: \n", te_pred)
print("要素数", len(te_pred))
# 各CATEの値のXの要素を示す
df_new = X.assign(te_pred=te_pred)
print("CATEを追加\n", df_new)
df_new.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_alldata_CATE_4th.csv")

# CATEの推定結果を確認
print("CATE of CausalForest: ", round(np.mean(te_pred), 2))

print("Percentile of CATE of CausalForest: 10%, 25%, 50%, 75%, 90%\n",
      np.quantile(a=te_pred, q=[.1, .25, .5, .75, .9]))

upper = np.quantile(a=te_pred, q=.9)  # CATE上位10％の境目
lower = np.quantile(a=te_pred, q=.1)  # CATE下位10％の境目
df_upper = df_new[(df_new["te_pred"] > upper)]  # CATE上位10%
df_lower = df_new[(df_new["te_pred"] < lower)]  # CATE下位10%
print("upper＝影響を受けやすかった10%: \n", df_upper)
print("lower＝影響を受けにくかった10%: \n", df_lower)

print("df_upper\n", df_upper.describe())
df_upper.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_upper_4th.csv")

print("df_lower\n", df_lower.describe())
df_lower.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_lower_4th.csv")

# CATE(全体)
# s.set()
s.displot(te_pred)
plt.savefig("/Volumes/Pegasus32R8/TTC/202211/cate_4th.svg")
# plt.show()

'''
# CATE(前半)
s.displot(te_pred_test1)
plt.show()

# CATE(後半)
s.displot(te_pred_test2)
plt.show()
'''


# https://towardsdatascience.com/causal-machine-learning-for-econometrics-causal-forests-5ab3aec825a7
# ★['Y0']['T0']問題！
plt.figure()
# calculate shap values of causal forest model
shap_values = causal_forest.shap_values(X)
# plot shap values
shap.summary_plot(shap_values['PLE_sum']['OCS_0or1'])


# Note that the structure of this estimator is based on the BaseEstimator and RegressorMixin from sklearn; however,
# here we predict treatment effects –which are unobservable– hence regular model validation and model selection
# techniques (e.g. cross validation grid search) do not work as we can never estimate a loss on a training sample,
# thus a tighter integration into the sklearn workflow is unlikely for now.
