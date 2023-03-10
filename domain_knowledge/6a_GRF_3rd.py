from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import seaborn as s
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
import shap
import sys
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

OCS_CUT_OFF = 12  # 強迫のCBCLカットライン。（-「項目数」点）以下は強迫なしとする。
"""
# impute前のデータフレーム
df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022domain/data4grf_before_imp.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# 特徴量 X、アウトカム Y、割り当て変数 T

# 3rd!全行がNaNの参加者は除外
df = df.dropna(subset=["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"],
               how='all')
print("PLE全部がNaNの物を削除\n", df)

# PLEの欠損値を補完
imputer = MissForest(criterion='squared_error', max_features=1.0)
df_imputed = imputer.fit_transform(df)
print("df_imputed\n", df_imputed)
df[df.columns.values] = df_imputed
df = df.round().astype(int)  # 各列を整数に丸める（身長、体重も丸め）

# PLEの合計点を作成(3rd!)
df_Y = df[["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"]]
print("df_Y\n", df_Y)
df["PLE_sum_3rd"] = df_Y.sum(axis=1)
print("3rdPLE合計\n", df["PLE_sum_3rd"])

df.to_csv("/Volumes/Pegasus32R8/TTC/2022domain/x_t_y_3rd.csv")
"""

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022domain/x_t_y_3rd.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")

Y = df["PLE_sum_3rd"]  # df['PLE_sum_3rd']  # 'CD60_1'(幻聴)とすると、単一項目で見られる
print("Y\n", Y)

# 第２期のAQ素点からAQを計算
# AQの合計点を作成
df_AQ = df[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]]
print(df_AQ)

for i in ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]:
    df_AQ = df_AQ.replace({i: {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}})
    print(df_AQ)

for i in ["BB125", "BB126", "BB127", "BB132"]:
    df_AQ = df_AQ.replace({i: {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}})
    print(df_AQ)

df["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df["AQ_sum"])

# 1つでも「1回以上あった」があった人をPLEありとする
df["bullied"] = 1
df["bullied"] = df["bullied"].where((df["AB61"] < 5) | (df["AD19"] < 5), 0)
print(df[["bullied", "AB61", "AD19"]])

# 第２期のOC
# 強迫の人数(cut off 5以上)
oc_2nd = df[["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116"]]  # 8項目ver
# oc_2nd = oc_ple[["BB39", "BB83"]]  # 2項目ver (2点以上は45人)
print("第２期のOC\n", oc_2nd)
print("NaN個数\n", oc_2nd.isnull().sum())
# oc_2nd = oc_2nd.dropna(how='any')  # ★強迫症状の項目にNaNを含むもの削除
oc_2nd["OCS_sum"] = oc_2nd.sum(axis=1)
print("第２期にOC欠損値なし\n", oc_2nd)

oc_pos = (oc_2nd["OCS_sum"] > OCS_CUT_OFF)
print("強迫症状カットオフ以上\n", oc_pos.sum())  # 5点以上だと115人

# OCS13以上を1、12以下を0にする
df["OCS_0or1"] = (oc_2nd["OCS_sum"] > OCS_CUT_OFF) * 1
print("OCS_0or1\n", df)  # 第２期にOC欠損値なしは2733行

# ★第1期にOCがない人を抽出★
oc_1st = df[["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"]]
print("第1期のOC\n", oc_1st)
print("NaN個数\n", oc_1st.isnull().sum())
# oc_1st = oc_1st.dropna(how='any')  # ★強迫症状の項目にNaNを含むもの削除
oc_1st["OCS_1st_sum"] = oc_1st.sum(axis=1)
print("第1期にOC欠損値なし\n", oc_1st)

df["OCS_1st"] = (oc_1st["OCS_1st_sum"] < OCS_CUT_OFF + 1) * 1
print("第１期でOCSが閾値以下の人\n", df)

# 第1期の強迫、PLEを除外
df = df.drop(["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"], axis=1)

T = df['OCS_0or1']  # 強迫CBCL5点以上であることをtreatmentとする
print("OCSあり: \n", T.sum())  # 41

# Y, Tを除外
X = df.drop(['PLE_sum_3rd', 'OCS_0or1'], axis=1)

X = X[["AA55", "AA58", "AB55", "AB58",
       "AB146", "AB161YOMI",
       "bullied",
       "TTC_sex", "AE1BMI", "AEIQ", "AAMedu", "AA79Fsep", "AB195",
       "AA127Respondent", "AQ_sum", "BR12",
       "AA110", "AB105", "AD36CPAQa_Imp", "AA165", "AA208", "AA189"]].copy()  # ★ドメイン知識で入れた項目と、探索的で入れた項目を
"""
# 第3期のPLEを除外
X = X.drop(["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"], axis=1)

# 第4期のPLEを除外
X = X.drop(["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)


# 第２期のAQ素点を除外
X = X.drop(["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"], axis=1)

# 第２期の強迫を除外
X = X.drop(["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116", "OCS_sum"], axis=1)
"""
print("X:\n", X)

# https://github.com/microsoft/EconML/blob/main/notebooks/Generalized%20Random%20Forests.ipynb
# 1. Causal Forest: Heterogeneous causal effects with no unobserved confounders

n_samples = len(df)
n_treatments = 1

Y_train, Y_val, T_train, T_val, X_train, X_val = train_test_split(Y, T, X, test_size=.2)
W = None

est = CausalForestDML(criterion='mse',
                      n_estimators=10000,
                      min_samples_leaf=10,
                      max_depth=None,
                      max_samples=0.5,
                      discrete_treatment=False,
                      honest=True,
                      inference=True,
                      cv=10,
                      model_t=RandomForestClassifier(max_depth=None,
                                                     max_features='sqrt',
                                                     min_samples_split=5,
                                                     min_samples_leaf=1,
                                                     n_estimators=1000,
                                                     n_jobs=int(cpu_count() / 2),
                                                     random_state=42),
                      model_y=RandomForestRegressor(max_depth=None,
                                                    max_features='sqrt',
                                                    # The number of features to consider when looking for the best split
                                                    min_samples_split=5,
                                                    min_samples_leaf=1,
                                                    n_estimators=2000,
                                                    n_jobs=int(cpu_count() / 2),
                                                    random_state=42),
                      n_jobs=int(cpu_count() / 2),
                      random_state=42)

# fit train data to causal forest model
est.fit(Y, T, X=X, W=W)

# Tが0→1になった時のYの変化量を予測
print("Calculate the average constant marginal CATE\n", est.const_marginal_ate(X))

# ATEを計算
print("Calculate the average treatment effect", est.ate(X, T0=0, T1=1))
lb0, ub0 = est.ate_interval(X, alpha=0.05)
print("ATE上限:", ub0)
print("ATE下限:", lb0)

# treatment effectを計算
te_pred = est.effect(X, T0=0, T1=1)
lb, ub = est.effect_interval(X, T0=0, T1=1, alpha=0.05)

# convert arrays to pandas dataframes for plotting
te_df = pd.DataFrame(te_pred, columns=['cate'])
lb_df = pd.DataFrame(lb, columns=['lb'])
ub_df = pd.DataFrame(ub, columns=['ub'])

print(te_df)

# merge dataframes and sort
df_plot = pd.concat([te_df, lb_df, ub_df], axis=1)
df_plot.sort_values('cate', inplace=True, ascending=True)
df_plot.reset_index(inplace=True, drop=True)

# calculate rolling mean
z = df_plot.rolling(window=30, center=True).mean()

# set plot size
fig, ax = plt.subplots(figsize=(12, 8))
# plot lines for treatment effects and confidence intervals
ax.plot(z['cate'],
        marker='.', linestyle='-', linewidth=0.5, label='CATE', color='indigo')
ax.plot(z['lb'],
        marker='.', linestyle='-', linewidth=0.5, color='steelblue')
ax.plot(z['ub'],
        marker='.', linestyle='-', linewidth=0.5, color='steelblue')
# label axes and create legend
ax.set_ylabel('Treatment Effects')
ax.set_xlabel('Number of observations (3rd)')
ax.legend()
# plt.show()

print("te_pred: \n", te_pred)
print("要素数", len(te_pred))
# 各CATEの値のXの要素を示す
df_new = df.assign(te_pred=te_pred)
print("CATEを追加_3rd\n", df_new)
# df_new.to_csv("test6.csv")

# CATEの推定結果を確認
print("CATE of CausalForest: ", np.mean(te_pred))

print("Percentile of CATE of CausalForest: 10%, 25%, 50%, 75%, 90%\n",
      np.quantile(a=te_pred, q=[.1, .25, .5, .75, .9]))

upper = np.quantile(a=te_pred, q=.95)  # CATE上位10％の境目
lower = np.quantile(a=te_pred, q=.05)  # CATE下位10％の境目
df_upper = df_new[(df_new["te_pred"] > upper)]  # CATE上位10%
df_lower = df_new[(df_new["te_pred"] < lower)]  # CATE下位10%
print("upper＝影響を受けやすかった5%: \n", df_upper)
print("lower＝影響を受けにくかった5%: \n", df_lower)

df_upper["group"] = 2
df_lower["group"] = 1

df_3rd = pd.concat([df_upper, df_lower])  # pd.merge(df1, df2, left_index=True, right_index=True)
print(df_3rd)
df_3rd.to_csv("/Volumes/Pegasus32R8/TTC/2022domain/upper_lower_3rd.csv")

# CATE(全体)
s.set()
s.displot(te_pred)
# plt.savefig("/Volumes/Pegasus32R8/TTC/202211/cate_4th.svg")
plt.show()
"""
# https://towardsdatascience.com/causal-machine-learning-for-econometrics-causal-forests-5ab3aec825a7
# ★['Y0']にはアウトカムを、['T0']にはtreatmentを入れる！
plt.figure()
# calculate shap values of causal forest model
shap_values = est.shap_values(X)
# plot shap values
shap.summary_plot(shap_values['PLE_sum_3rd']['OCS_0or1'], max_display=len(X.columns))
"""
# Note that the structure of this estimator is based on the BaseEstimator and RegressorMixin from sklearn; however,
# here we predict treatment effects –which are unobservable– hence regular model validation and model selection
# techniques (e.g. cross validation grid search) do not work as we can never estimate a loss on a training sample,
# thus a tighter integration into the sklearn workflow is unlikely for now.
