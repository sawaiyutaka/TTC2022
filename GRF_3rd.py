import pandas as pd
import numpy as np
import seaborn as s
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
import shap

# imputeした後のデータフレーム、PLEとAQの合計得点前
df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/X_T_Y_imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# 特徴量 X、アウトカム Y、割り当て変数 T
Y = df['PLE_sum_3rd']  # 'CD65_1'などとすると、単一項目で見られる
print("Y\n", Y)

T = df['OCS_0or1']  # 強迫CMCL5点以上であることをtreatmentとする
print("T\n", T)

# 第1期の強迫、PLEを除外したXを読み込み
X = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/X_imputed.csv", delimiter=",")
X = X.set_index("SAMPLENUMBER")
print("X:\n", X)
print("補完後のNaN個数\n", X.isnull().sum())

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
                                                     n_estimators=10000,
                                                     n_jobs=15,
                                                     # number of jobs to run in parallel(-1 means using all processors)
                                                     random_state=2525),  # LassoCV(max_iter=100000),
                      model_y=RandomForestRegressor(max_depth=None,
                                                    max_features=100,
                                                    # The number of features to consider when looking for the best split
                                                    min_samples_split=5,
                                                    min_samples_leaf=1,
                                                    n_estimators=10000,
                                                    n_jobs=15,
                                                    random_state=2525),  # LassoCV(max_iter=100000),
                      random_state=2525,
                      n_jobs=15)

# fit train data to causal forest model
est.fit(Y, T, X=X, W=W)

# Tが0→1になった時のYの変化量を予測
print("Calculate the average constant marginal CATE\n", est.const_marginal_ate(X))

# ATEを計算
print("Calculate the average treatment effect", est.ate(X, T0=0, T1=1))
lb0, ub0 = est.ate_interval(X, alpha=0.05)
print("ATE上限:", ub0)
print("ATE下限:", lb0)

# feature importance
print("covariate\n", list(X.columns.values))
covariate = list(X.columns.values)
print("feature_importance\n", list(est.feature_importances_))
feature_importance = list(est.feature_importances_)

print([covariate, feature_importance])

lst = [covariate, feature_importance]
df1 = pd.DataFrame(lst, index=['covariate', 'feature_importance'])
df2 = df1.T
print(df2)

df2.sort_values('feature_importance', inplace=True, ascending=False)
print(df2)

df2.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/importance_3rd_sort.csv")

'''
# 半分に分割してテスト
# test1
X_test1 = X.iloc[:int(n_samples / 2), :]
# test2
X_test2 = X.iloc[int(n_samples / 2):n_samples, :]

print("X_test1: \n", X_test1)
print("X_test2: \n", X_test2)
'''
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

'''
# X_testのみでCATEを計算
te_pred_test1 = est.effect(X_test1)
te_pred_test2 = est.effect(X_test2)
'''

print("te_pred: \n", te_pred)
print("要素数", len(te_pred))
# 各CATEの値のXの要素を示す
df_new = df.assign(te_pred=te_pred)
print("CATEを追加_3rd\n", df_new)
df_new.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_alldata_CATE_3rd.csv")

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

all_1st = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_1st_outcome_Imp.csv",
                        delimiter=",", low_memory=False)
all_1st = all_1st.set_index("SAMPLENUMBER")

print("df_upper_3rd\n", df_upper.describe())
cols_to_use = all_1st.columns.difference(df_upper.columns)

print("第１期量的データにあって、upper, lowerに含まれない項目を検出\n", cols_to_use)
upper = df_upper.join([all_1st[cols_to_use]], how='inner')
print("df_upper_3rd\n", upper.describe())
upper.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_upper_3rd.csv")

print("df_lower_3rd\n", df_lower.describe())
lower = df_lower.join([all_1st[cols_to_use]], how='inner')
print("df_lower_3rd\n", lower.describe())
lower.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_outcome/TTC2022_lower_3rd.csv")

# CATE(全体)
s.set()
s.displot(te_pred)
# plt.savefig("/Volumes/Pegasus32R8/TTC/202211/cate_4th.svg")
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
shap_values = est.shap_values(X)
# plot shap values
shap.summary_plot(shap_values['PLE_sum_3rd']['OCS_0or1'])  # , max_display=30, order=shap_values.abs.max(0))

# Note that the structure of this estimator is based on the BaseEstimator and RegressorMixin from sklearn; however,
# here we predict treatment effects –which are unobservable– hence regular model validation and model selection
# techniques (e.g. cross validation grid search) do not work as we can never estimate a loss on a training sample,
# thus a tighter integration into the sklearn workflow is unlikely for now.
