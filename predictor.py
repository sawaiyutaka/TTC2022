import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# imputeした後のデータフレーム、PLEとAQの合計得点前
df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/base_ple_imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# PLEの合計点を作成(第3期)
df_Y = df[["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"]]
print("df_Y\n", df_Y)
df_Y["PLE_sum"] = df_Y.sum(axis=1)
print("第3回PLE合計\n", df_Y["PLE_sum"])

"""
# PLEの合計点を作成(第4期)
df_Y = df[["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"]]
print("df_Y\n", df_Y)
df_Y["PLE_sum"] = df_Y.sum(axis=1)
print("第4回PLE合計\n", df_Y["PLE_sum"])
"""

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

# Xから回答日、回答時点の月齢は削除
X = df.drop(["AA1YEAR", "AA1MONTH", "AA1DAY", "AA1age"], axis=1)

# 第3期のPLEを除外
X = X.drop(["PLE_sum", "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"],
           axis=1)

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

# X→Yの予測に使うランダムフォレストのグリッドサーチ
# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model_y = RandomForestRegressor(max_depth=None,
                                max_features=200,
                                # The number of features to consider when looking for the best split
                                # 'sqrt'も可能
                                min_samples_split=5,
                                min_samples_leaf=1,
                                n_estimators=100000,
                                n_jobs=25,  # number of jobs to run in parallel(-1 means using all processors)
                                random_state=2525)
model_y.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model_y.predict(X_val)
print("model_score: ", model_y.score(X_val, Y_val))

# ハイパーパラメータをチューニング
search_params = {
    'n_estimators': [10000, 50000, 100000, 200000],
    'max_features': [i for i in range(2, int(X_train.shape[1] / 2), 50)],
    'random_state': [2525],
    # 'min_samples_split': [2, 5, 10, 20],
    # 'min_samples_leaf': [1, 5, 10, 20]
    # 'max_depth': [20, 30, 40]
    # RandomForestRegressor()
}

gsr = GridSearchCV(
    RandomForestRegressor(),
    search_params,
    cv=3,
    n_jobs=25,
    verbose=True
)

gsr.fit(X_train, Y_train)

# 最もよかったモデル
print(gsr.best_estimator_)
print("最もよかったモデルの評価", gsr.best_estimator_.score(X_val, Y_val))


# X→Tの予測に使うランダムフォレストのグリッドサーチ
# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model_t = RandomForestClassifier(max_depth=None,
                                 max_features='sqrt',
                                 # The number of features to consider when looking for the best split
                                 # 'sqrt'も可能
                                 min_samples_split=5,
                                 min_samples_leaf=1,
                                 n_estimators=100000,
                                 n_jobs=25,  # number of jobs to run in parallel(-1 means using all processors)
                                 random_state=2525)

model_t.fit(X_train, T_train)

# 学習済みモデルの評価
predicted_T_val = model_t.predict(X_val)
print("model_score: ", model_t.score(X_val, T_val))

# X→Yの予測に使うランダムフォレストのグリッドサーチ
# ハイパーパラメータをチューニング
search_params = {
    'n_estimators': [10000, 50000, 100000, 200000],
    'max_features': [i for i in range(2, int(X_train.shape[1] / 2), 50)],
    'random_state': [2525],
    # 'min_samples_split': [2, 5, 10, 20],
    # 'min_samples_leaf': [1, 5, 10, 20]
    # 'max_depth': [20, 30, 40]
    # RandomForestRegressor()
}

gsr = GridSearchCV(
    RandomForestClassifier(),
    search_params,
    cv=3,
    n_jobs=25,
    verbose=True
)

gsr.fit(X_train, T_train)

# 最もよかったモデル
print(gsr.best_estimator_)
print("最もよかったモデルの評価", gsr.best_estimator_.score(X_val, T_val))
