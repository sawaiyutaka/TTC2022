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

OCS_CUT_OFF = 1  # 強迫のCBCLカットライン。（-「項目数」点）以下は強迫なしとする。

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2023retry/df4grf_xty.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")

# PLEの合計点を作成
df_Y = df[["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
           "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1"]]

print("df_Y\n", df_Y)
df["PLE_sum"] = df_Y.sum(axis=1)
print("PLE合計\n", df["PLE_sum"])

Y = df["PLE_sum"]  # 'CD60_1'(幻聴)とすると、単一項目で見られる
print("Y\n", Y)

T = df['OCS_0or1']
print("OCSあり: \n", T.sum())  # 人

# Xを指定
X = df[["AA55", "AA58", "AB55", "AB58",
        "AB146", "AB161YOMI",
        "bullied",
        "TTC_sex", "AE1BMI", "AEIQ", "AB161MIQ", "AA79Fsep", "SES",
        "AA127Respondent", "AQ_sum", "BR12",
        # exploratoryで抽出された項目
        "AA110", "AB105",  # "AD36CPAQa_Imp", "AD36CPAQm_Imp", "AD36CPAQf_Imp",
        "AA165", "AA208", "AA189",
        "AD52", "AD53", "AD54", "AD55", "AD56"  # CPAQのMF尺度は合計得点を研究で使うことが少ない
        ]].copy()  # ★ドメイン知識で入れた項目と、探索的で入れた項目を

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
                      n_jobs=int(cpu_count() *4 / 5),
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
ax.set_xlabel('Number of observations')
ax.legend()
# plt.show()

print("te_pred: \n", te_pred)
print("要素数", len(te_pred))
# 各CATEの値のXの要素を示す
df_new = X.assign(T=T, Y=Y, te_pred=te_pred)
print("CATEを追加\n", df_new)

# CATEの推定結果を確認
print("CATE of CausalForest: ", np.mean(te_pred))

print("Percentile of CATE of CausalForest: 10%, 25%, 50%, 75%, 90%\n",
      np.quantile(a=te_pred, q=[.1, .25, .5, .75, .9]))

upper = np.quantile(a=te_pred, q=.9)  # CATE上位10％の境目
lower = np.quantile(a=te_pred, q=.1)  # CATE下位10％の境目
df_upper = df_new[(df_new["te_pred"] > upper)]  # CATE上位10%
df_lower = df_new[(df_new["te_pred"] < lower)]  # CATE下位10%
print("upper＝影響を受けやすかった10%: \n", df_upper)
print("lower＝影響を受けにくかった10%: \n", df_lower)

df_upper["group"] = 2
df_lower["group"] = 1

df_concat = pd.concat([df_upper, df_lower])
print(df_concat)
df_concat.to_csv("/Volumes/Pegasus32R8/TTC/2023retry/upper_lower_combined.csv")

# CATE(全体)
s.set()
s.displot(te_pred)
plt.show()
