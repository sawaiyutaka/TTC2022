import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
import lightgbm as lgb
from IPython.core.display_functions import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def True_Pred_map(pred_df):
    RMSE = np.sqrt(mean_squared_error(pred_df['true'].tolist(), pred_df['pred'].tolist()))
    R2 = r2_score(pred_df['true'], pred_df['pred'])
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.scatter('true', 'pred', data=pred_df)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Pred Value', fontsize=15)
    ax.set_xlim(pred_df.min().min() - 0.1, pred_df.max().max() + 0.1)
    ax.set_ylim(pred_df.min().min() - 0.1, pred_df.max().max() + 0.1)
    x = np.linspace(pred_df.min().min() - 0.1, pred_df.max().max() + 0.1, 2)
    y = x
    ax.plot(x, y, 'r-')
    plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 5))), transform=ax.transAxes, fontsize=15)
    plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)
    plt.show()


df = pd.read_table("3rd_X_T_Y.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

#
df = df[df["PLE_sum_3rd"] > 9].copy()

Y = df["PLE_sum_3rd"]  # df['PLE_sum_3rd']  # 'CD65_1'などとすると、単一項目で見られる
print("Y\n", Y)
# s.displot(Y)
# plt.show()
total = df.shape[0]
Y_count = df[Y > 9].shape[0]

print('トータル：{}'.format(total))
print('10以上の被説明変数：{}'.format(Y_count))

# Y, Tを除外
X = df.drop(['PLE_sum_3rd', 'OCS_0or1'], axis=1)

# 第3期のPLEを除外
X = X.drop(["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"], axis=1)
# 第4期のPLEを除外
X = X.drop(["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)
print("X:\n", X)

Y_train, Y_test, X_train, X_test = train_test_split(Y, X, test_size=.2, random_state=42)

model_y = RandomForestRegressor(max_depth=None,
                                max_features='sqrt',
                                # The number of features to consider when looking for the best split
                                # 'sqrt'も可能
                                min_samples_split=5,
                                min_samples_leaf=1,
                                n_estimators=1000,
                                # n_jobs=25,  # number of jobs to run in parallel(-1 means using all processors)
                                random_state=2525)
model_y.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model_y.predict(X_test)
print("model_score: ", model_y.score(X_test, Y_test))

"""
model = sm.ZeroInflatedPoisson(endog=Y,
                               exog=sm.add_constant(X),
                               exog_infl=sm.add_constant(X),
                               inflation='logit')
result = model.fit_regularized()
print('AIC:{}'.format(result.aic.round()))
display(result.summary())
"""

out = sm.ZeroInflatedPoisson(Y_train, X_train, exog_infl=X_train)
res = out.fit()
y_test_pred = res.predict(X_test, exog_infl=X_test)
print(res.summary())
plt.clf()
plt.hist([Y_test, y_test_pred], log=True, bins=max(Y_test))
plt.legend(('orig', 'pred'))
plt.show()

pred_df = pd.concat([Y, y_test_pred], axis=1)
print(pred_df)
pred_df = pred_df.fillna(9)
print(pred_df)
pred_df = pred_df.set_axis(['true', 'pred'], axis='columns')
print(pred_df)
True_Pred_map(pred_df)



lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test)

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test)

params = {'metric': 'rmse',
          'max_depth': 9
          }

gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=50,
                )

predicted = gbm.predict(X_test)

# 関数の処理で必要なライブラリ
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 予測値と正解値を描写する関数
def True_Pred_map(pred_df):
    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))
    R2 = r2_score(pred_df['true'], pred_df['pred'])
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.scatter('true', 'pred', data=pred_df)
    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Pred Value', fontsize=15)
    ax.set_xlim(pred_df.min().min() - 0.1, pred_df.max().max() + 0.1)
    ax.set_ylim(pred_df.min().min() - 0.1, pred_df.max().max() + 0.1)
    x = np.linspace(pred_df.min().min() - 0.1, pred_df.max().max() + 0.1, 2)
    y = x
    ax.plot(x, y, 'r-')
    plt.text(0.1, 0.9, 'RMSE = {}'.format(str(round(RMSE, 5))), transform=ax.transAxes, fontsize=15)
    plt.text(0.1, 0.8, 'R^2 = {}'.format(str(round(R2, 5))), transform=ax.transAxes, fontsize=15)
    plt.show()


pred_df = pd.concat([Y_test.reset_index(drop=True), pd.Series(predicted)], axis=1)
pred_df.columns = ['true', 'pred']

print(pred_df.head())

True_Pred_map(pred_df)

lgb.plot_importance(gbm, height=0.5, figsize=(8, 16))
plt.show()


"""
RMSE_list = []
count = []
for i in range(1, 15):
    params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'metric': 'rmse',
              'max_depth': i}

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=50)

    predicted = gbm.predict(X_test)
    pred_df = pd.concat([Y_test.reset_index(drop=True), pd.Series(predicted)], axis=1)
    pred_df.columns = ['true', 'pred']
    RMSE = np.sqrt(mean_squared_error(pred_df['true'], pred_df['pred']))
    RMSE_list.append(RMSE)
    count.append(i)

plt.figure(figsize=(16,8))
plt.plot(count, RMSE_list, marker="o")
plt.title("RMSE Values", fontsize=30)
plt.xlabel("max_depth", fontsize=20)
plt.ylabel("RMSE Value", fontsize=20)
plt.grid(True)
"""
