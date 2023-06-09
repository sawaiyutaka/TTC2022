from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

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
X = df[["TTC_sex", "AE1BMI", "AEIQ", "AB161MIQ", "AA127Respondent", "SES", "bullied", "AQ_sum",
        # exploratoryで抽出された項目
        ]].copy()  # ★ドメイン知識で入れた項目と、探索的で入れた項目を
print("X:\n", X)

Y_train, Y_test, T_train, T_test, X_train, X_test = train_test_split(Y, T, X, test_size=.2)


# Yの予測に使うランダムフォレストのグリッドサーチ
param_grid_y = {
    'n_estimators': list(range(200, 2001, 200)),
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt'],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    'random_state': [0]
}
model_y = RandomForestRegressor()

# グリッドサーチの実行
grid_search_y = GridSearchCV(estimator=model_y, param_grid=param_grid_y, cv=5, scoring='neg_mean_squared_error')
grid_search_y.fit(X_train, Y_train)

# 最適なハイパーパラメータの結果の表示
print("Best parameters:", grid_search_y.best_params_)
print("Best RMSE score:", np.sqrt(-grid_search_y.best_score_))
# Best parameters: {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 0}
# Best RMSE score: 2.4018503294769324
# Test RMSE: 2.244629262807043
# Test R^2: 0.012591012461121487

# テストデータでの予測と評価
best_model = grid_search_y.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)
print("Test RMSE:", rmse)
print("Test R^2:", r2)


# Tの予測に使うランダムフォレストのグリッドサーチ
param_grid_t = {
    'n_estimators': list(range(200, 2001, 200)),
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt'],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    'random_state': [0]
}

# ランダムフォレストモデルのインスタンス化
model_t = RandomForestClassifier()

# グリッドサーチの実行
grid_search_t = GridSearchCV(estimator=model_t, param_grid=param_grid_t, cv=5, scoring='accuracy')
grid_search_t.fit(X_train, Y_train)

# 最適なハイパーパラメータの結果の表示
print("Best parameters:", grid_search_t.best_params_)
print("Best accuracy score:", grid_search_t.best_score_)
# Best parameters: {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 0}
# Best accuracy score: 0.6165383476458395

# テストデータでの予測と評価
best_model = grid_search_t.best_estimator_
T_pred = best_model.predict(X_test)
accuracy = accuracy_score(T_test, T_pred)
precision = precision_score(T_test, T_pred)
recall = recall_score(T_test, T_pred)
f1 = f1_score(T_test, T_pred)
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 score:", f1)

