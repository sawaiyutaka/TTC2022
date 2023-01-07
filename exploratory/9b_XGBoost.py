import sys
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import pandas as pd
import numpy as np
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt
from missingpy import MissForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
import shap
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from multiprocessing import cpu_count
from dcekit.variable_selection import search_high_rate_of_same_values, search_highly_correlated_variables

df = pd.read_table("df_3rd.csv", delimiter=",")
df = df.set_index("SAMPLENUMBER")
print(df)

y = df["group_3rd"]
print(y)

X = df.drop(["group_3rd"], axis=1)

# 参照！！：https://datadriven-rnd.com/2021-02-03-231858/

# 分散が０の変数削除
del_num1 = np.where(X.var() == 0)
X = X.drop(X.columns[del_num1], axis=1)

# 変数選択（互いに相関関係にある変数の内一方を削除）
threshold_of_r = 0.95  # 変数選択するときの相関係数の絶対値の閾値
corr_var = search_highly_correlated_variables(X, threshold_of_r)
X.drop(X.columns[corr_var], axis=1, inplace=True)

"""
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

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state


class BorutaPyForLGB(BorutaPy):
    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=0):
        super().__init__(estimator, n_estimators, perc, alpha,
                         two_step, max_iter, random_state, verbose)
        if random_state is None:
            self.random_state_input = np.random.randint(0, 2 ** 64 - 1)
        elif isinstance(random_state, int):
            self.random_state_input = random_state
        else:
            raise TypeError('random_state must be int or None')

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if (depth == None) or (depth <= 0):
            depth = 10
        f_repr = 100
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)
        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            self.estimator.set_params(random_state=self.random_state_input)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)

            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        return self


"""XGBoost の scikit-learn インターフェースを使ったサンプルコード (二値分類)"""

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=42,
                                                    stratify=y)
# データ全体で学習
model = lgb.LGBMClassifier(objective='binary',
                           num_leaves=23,
                           learning_rate=0.1,
                           n_estimators=1000)
model.fit(X_train.values, y_train.values)

y_test_pred = model.predict(X_test.values)
print(confusion_matrix(y_test.values, y_test_pred, labels=model.classes_), '\n')
print('SCORE with ALL Features: %1.2f\n' % accuracy_score(y_test, y_test_pred))

# Borutaで特徴量選択 (一部書き換えたBorutaPyを使います)
feat_selector = BorutaPyForLGB(model,
                               n_estimators='auto',  # 特徴量の数に比例して、木の本数を増やす
                               verbose=2,
                               alpha=0.05,  # 有意水準
                               max_iter=100,  # 試行回数
                               perc=80,  # perc,  # ランダム生成変数の重要度の何％を基準とするか
                               two_step=False,  # two_stepがない方、つまりBonferroniを用いたほうがうまくいく
                               random_state=0
                               )
feat_selector.fit(X_train.values, y_train.values)
print(X_train.columns[feat_selector.support_])

# 選択したFeatureを取り出し
X_train_selected = X_train.iloc[:, feat_selector.support_]
X_test_selected = X_test.iloc[:, feat_selector.support_]
print(X_test_selected.head())

# 選択したFeatureで学習
model = lgb.LGBMClassifier(objective='binary',
                           num_leaves=23,
                           learning_rate=0.1,
                           n_estimators=1000, )
model.fit(X_train_selected.values, y_train.values)

y_test_pred = model.predict(X_test_selected.values)
print(confusion_matrix(y_test.values, y_test_pred, labels=model.classes_), '\n')
print('SCORE with selected Features: %1.2f\n' % accuracy_score(y_test, y_test_pred))

explainer = shap.TreeExplainer(model=model, feature_perturbation='tree_path_dependent', model_output='margin')
shap_values = explainer.shap_values(X=X_test_selected)
shap.summary_plot(shap_values, X_test_selected)