import sys

import shap
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count

# from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables

df = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/imputed.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

# 第２期のAQ素点からAQを計算
# AQの合計点を作成
df_AQ = df[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]].copy()
print(df_AQ)

for i in ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]:
    df_AQ = df_AQ.replace({i: {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}})
    print(df_AQ)

for i in ["BB125", "BB126", "BB127", "BB132"]:
    df_AQ = df_AQ.replace({i: {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}})
    print(df_AQ)

df["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df["AQ_sum"])
df = df.drop(["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"], axis=1)
df = df.drop(columns=["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"])
df = df.drop(columns=["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"])
df = df.drop(columns=["BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116", "OCS_sum"])

y = df["OCS_0or1"]
print(y)

X = df.drop(["OCS_0or1"], axis=1)

# 参照！！：https://datadriven-rnd.com/2021-02-03-231858/
"""
# 分散が０の変数削除
del_num1 = np.where(X.var() == 0)
X = X.drop(X.columns[del_num1], axis=1)

# 変数選択（互いに相関関係にある変数の内一方を削除）
threshold_of_r = 0.95  # 変数選択するときの相関係数の絶対値の閾値
corr_var = search_highly_correlated_variables(X, threshold_of_r)
X.drop(X.columns[corr_var], axis=1, inplace=True)

# 同じ値を多くもつ変数の削除
inner_fold_number = 2  # CVでの分割数（予定）
rate_of_same_value = []
threshold_of_rate_of_same_value = (inner_fold_number - 1) / inner_fold_number
for col in X.columns:
    same_value_number = X[col].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / X.shape[0]))
del_var_num = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)
X.drop(X.columns[del_var_num], axis=1, inplace=True)
"""
print(X.shape)
print(X.head())

Y_train, Y_test, X_train, X_test = train_test_split(y, X, test_size=0.3, stratify=y)

rf = RandomForestClassifier(
    n_estimators=5000,
    random_state=42,
    n_jobs=int(cpu_count() / 2),
    max_depth=7,
    max_features='sqrt'
)
rf.fit(X_train.values, Y_train.values)
print(rf.classes_)
print(confusion_matrix(Y_test.values, rf.predict(X_test.values), labels=rf.classes_))
print("before boruta\n", accuracy_score(Y_test, rf.predict(X_test)))

"""
# pパーセンタイルの最適化
corr_list = []
for n in range(10000):
    shadow_features = np.random.rand(X_train.shape[0]).T
    corr = np.corrcoef(X_train, shadow_features, rowvar=False)[-1]
    corr = abs(corr[corr < 0.95])
    corr_list.append(corr.max())

corr_array = np.array(corr_list)
perc = 100 * (1 - corr_array.max())
print('pパーセンタイル:', round(perc, 2))
"""

# Borutaの実施
feat_selector = BorutaPy(rf,
                         n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                         verbose=2,
                         alpha=0.05,  # 有意水準
                         max_iter=100,  # 試行回数
                         perc=100,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
                         two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
                         random_state=0
                         )

# データの二度漬けになるので特徴量選択する際にもtestを含めてはいけない
feat_selector.fit(X_train.values, Y_train.values)
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_test_selected = X_test.iloc[:, feat_selector.support_]
print('boruta後の変数の数:', X_train_selected.shape[1])

rf2 = RandomForestClassifier(
    n_estimators=5000,
    random_state=42,
    n_jobs=int(cpu_count() / 2),
    max_features='sqrt'
)
rf2.fit(X_train_selected.values, Y_train.values)

print(rf2.classes_)
print(confusion_matrix(Y_test.values, rf2.predict(X_test_selected.values), labels=rf.classes_))
print("after boruta\n", accuracy_score(Y_test, rf2.predict(X_test_selected)))

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(rf2.predict, X_test_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test_selected)

shap.plots.bar(shap_values, max_display=len(X_test_selected.columns))
shap.summary_plot(shap_values, max_display=len(X_test_selected.columns))

"""
# 学習データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X_selected, y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

# 学習データを、学習用と検証用に分ける
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y_train)

# データを格納する
# 学習用
xgb_train = xgb.DMatrix(X_train, label=y_train)

# 検証用
xgb_eval = xgb.DMatrix(X_eval, label=y_eval)

# テスト用
xgb_test = xgb.DMatrix(X_test, label=y_test)

# 学習用のパラメータ
xgb_params = {
    # 二値分類問題
    'objective': 'binary:logistic',
    # 評価指標
    'eval_metric': 'logloss',
}
evals = [(xgb_train, 'train'), (xgb_eval, 'eval')] # 学習に用いる検証用データ
evaluation_results = {}                            # 学習の経過を保存する箱
bst = xgb.train(xgb_params,                        # 上記で設定したパラメータ
                xgb_train,                         # 使用するデータセット
                num_boost_round=1000,              # 学習の回数
                early_stopping_rounds=10,          # アーリーストッピング
                evals=evals,                       # 学習経過で表示する名称
                evals_result=evaluation_results,   # 上記で設定した検証用データ
                verbose_eval=10                    # 学習の経過の表示(10回毎)
                )

# 検証用データが各クラスに分類される確率を計算する
y_pred_proba = bst.predict(xgb_test)
# しきい値 0.5 で 0, 1 に丸める
y_pred = np.where(y_pred_proba > 0.5, 1, 0)

# 精度 (Accuracy) を検証する
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)

print(confusion_matrix(y_test, y_pred))

# feature importanceを表示
xgb.plot_importance(bst)

# 学習過程の可視化
plt.figure()
plt.plot(evaluation_results['train']['logloss'], label='train')
plt.plot(evaluation_results['eval']['logloss'], label='eval')
plt.ylabel('Log loss')
plt.xlabel('Boosting round')
plt.title('Training performance')
plt.legend()

plt.figure()
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(xgb_test)
plt.show()

rf2 = RandomForestClassifier(
    n_estimators=5000,
    random_state=42,
    n_jobs=int(cpu_count() / 2),
    max_features='sqrt'
)
rf2.fit(X_train_selected.values, Y_train.values)

print(rf2.classes_)
print(confusion_matrix(Y_test.values, rf2.predict(X_test_selected.values), labels=rf.classes_))
print("after boruta\n", accuracy_score(Y_test, rf2.predict(X_test_selected)))

# sklearnの機械学習モデル（ランダムフォレスト）のインスタンスを作成する
# 教師データと教師ラベルを使い、fitメソッドでモデルを学習
model = RandomForestRegressor(max_depth=None,
                              max_features='sqrt',
                              # X_train.shape[1],  # The number of features to consider when looking for the best split
                              # 'sqrt'も可能
                              min_samples_split=5,
                              min_samples_leaf=1,
                              n_estimators=5000,
                              n_jobs=int(cpu_count() / 2),
                              random_state=42)
model.fit(X_train, Y_train)

# 学習済みモデルの評価
predicted_Y_val = model.predict(X_test)
print("model_score: ", model.score(X_test, Y_test))

# Borutaを実行
rf = RandomForestRegressor(n_jobs=int(cpu_count() / 2), max_depth=7)
feat_selector = BorutaPy(rf, n_estimators='auto', two_step=False, verbose=2, random_state=42)
feat_selector.fit(X_train.values, Y_train.values)
print(X_train.columns[feat_selector.support_])

# 選択したFeatureを取り出し
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_val_selected = X_test.iloc[:, feat_selector.support_]
print(X_val_selected.head())

# 選択したFeatureで学習
rf2 = RandomForestRegressor(max_depth=None,
                            max_features='auto',
                            # X_train.shape[1],  # The number of features to consider when looking for the best split
                            # 'sqrt'も可能
                            min_samples_split=5,
                            min_samples_leaf=1,
                            n_estimators=5000,
                            n_jobs=int(cpu_count() / 2),
                            random_state=42)
rf2.fit(X_train_selected.values, Y_train.values)

predicted_Y_val_selected = rf2.predict(X_val_selected.values)
print("model_score_2: ", rf2.score(X_val_selected, Y_test))

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X)

# shap valueで評価（時間がかかる）
# Fits the explainer
explainer = shap.Explainer(y_pred, X_test_selected)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test_selected)

# shap.plots.bar(shap_values, max_display=len(X_test_selected.columns))
"""
