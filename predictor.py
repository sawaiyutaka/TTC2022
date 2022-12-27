import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# imputeした後のデータフレーム、PLEとAQの合計得点前
df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_outcome/X_T_Y_imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)


# 特徴量 X、アウトカム Y、割り当て変数 T
Y = df['PLE_sum_3rd']  # 'CD65_1'などとすると、単一項目で見られる
# df_Y = df.replace({'CD65_1': {1: 0, 2: 0, 3: 0, 4: 1}})  # 幻聴の有無
# Y = df_Y['CD65_1']  # 感度分析としてYを二値にした場合にどうなるか
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

"""
# Yを二値にした場合
model_y = RandomForestClassifier(max_depth=None,
                                 max_features='sqrt',
                                 # The number of features to consider when looking for the best split
                                 # 'sqrt'も可能
                                 min_samples_split=5,
                                 min_samples_leaf=1,
                                 n_estimators=1000,
                                 n_jobs=20,  # number of jobs to run in parallel(-1 means using all processors)
                                 random_state=2525)

model_y.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model_y.predict(X_val)
print("model_score: ", model_y.score(X_val, Y_val))
"""



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
"""
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
"""
# Yの予測に重要なパラメータを探す
explainer = shap.Explainer(model_y.predict, X_val)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_val, max_evals=1000)
# max_evals=500 is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = 887!

shap.plots.bar(shap_values, max_display=30)
shap.summary_plot(shap_values, max_display=30)

'''
# X→Tの予測に使うランダムフォレストのグリッドサーチ
# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model_t = RandomForestClassifier(max_depth=None,
                                 max_features='sqrt',
                                 # The number of features to consider when looking for the best split
                                 # 'sqrt'も可能
                                 min_samples_split=5,
                                 min_samples_leaf=1,
                                 n_estimators=10000,
                                 n_jobs=25,  # number of jobs to run in parallel(-1 means using all processors)
                                 random_state=2525)

model_t.fit(X_train, T_train)

# 学習済みモデルの評価
predicted_T_val = model_t.predict(X_val)
print("model_score: ", model_t.score(X_val, T_val))

# X→Yの予測に使うランダムフォレストのグリッドサーチ
# ハイパーパラメータをチューニング
search_params = {
    'n_estimators': [10000, 50000, 20000],
    'max_features': [i for i in range(2, int(X_train.shape[1] / 3), 50)],
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
'''