"""
欠損フラグ導入（missing indicator）→ Boruta（XGBoost推定器）で特徴量選択 →
選択特徴量でXGBoost学習 → outer CVでAUC評価 → 選択頻度を集計して保存

前提：
- /Volumes/Pegasus32R8/TTC/2025thesis/X_NAN_under_5percent_and_Y.csv を読み込む
- social_cohesion / atopy / AQ_sum / bullied を作成（あなたのコード踏襲）
- OCS_0or1 == 1 に絞る（あなたのコード踏襲。必要ならOFFにしてください）
- PE（14/16歳）を PLE>2 を1つでも含む→1、全て<2→0 で定義し、
  それ以外（欠損や混在）は除外（あなたの定義思想を踏襲）

出力：
- outer CV foldごとのAUCなど: sens_boruta_xgb_missingflag_outer_results.csv
- 特徴量選択頻度: sens_boruta_xgb_missingflag_feature_frequency.csv
- 各foldで選ばれた特徴量リスト（行ごと）: sens_boruta_xgb_missingflag_selected_lists.csv
"""

from multiprocessing import cpu_count
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb

from boruta import BorutaPy
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer, confusion_matrix


# =========================
# 0) 設定
# =========================
DATA_PATH = "/Volumes/Pegasus32R8/TTC/2025thesis/X_NAN_under_5percent_and_Y.csv"

OUT_OUTER = "/Volumes/Pegasus32R8/TTC/2025thesis/sens_boruta_xgb_missingflag_outer_results.csv"
OUT_FREQ  = "/Volumes/Pegasus32R8/TTC/2025thesis/sens_boruta_xgb_missingflag_feature_frequency.csv"
OUT_LISTS = "/Volumes/Pegasus32R8/TTC/2025thesis/sens_boruta_xgb_missingflag_selected_lists.csv"

FILTER_OCS_POSITIVE_ONLY = True   # あなたのコードに合わせる（OCS_0or1==1に絞る）
REPEATS = 20                      # まずは20で動作確認推奨。最終的に100などに増やす
OUTER_SPLITS = 4
INNER_SPLITS = 4
RANDOM_STATE = 42


# =========================
# 1) 関数
# =========================
def youden_index_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    return sensitivity + specificity - 1


def add_missing_indicators_and_fill0(X: pd.DataFrame):
    """
    欠損フラグ（_missing）を作り、元の欠損は0で埋める。
    BorutaはNaNを受け付けないため必須。
    """
    X2 = X.copy()
    miss_cols = []

    for col in X.columns:
        miss = X2[col].isna().astype(int)
        if miss.sum() > 0:
            X2[col + "_missing"] = miss
            miss_cols.append(col)
            X2[col] = X2[col].fillna(0)

    # 念のため：残存NaNチェック
    n_nan = int(X2.isna().sum().sum())
    if n_nan != 0:
        # object列などでNaNが残る可能性があるので強制処理
        X2 = X2.apply(pd.to_numeric, errors="coerce")
        X2 = X2.fillna(0)
        n_nan2 = int(X2.isna().sum().sum())
        if n_nan2 != 0:
            raise ValueError(f"NaNが残っています: {n_nan2}")

    return X2, miss_cols


# =========================
# 2) データ読み込み
# =========================
df = pd.read_table(DATA_PATH, delimiter=",").set_index("SAMPLENUMBER")
print("Loaded df:", df.shape)


# =========================
# 3) 特徴量作成（あなたのコード踏襲）
# =========================
# social cohesion
social_cols = ["AA57", "AA58", "AA59", "AA60", "AA61"]
if set(social_cols).issubset(df.columns):
    df["social_cohesion"] = df[social_cols].sum(axis=1)

# atopy
if set(["AF37", "AF38"]).issubset(df.columns):
    df["atopy"] = ((df[["AF37", "AF38"]] == 1).any(axis=1)).astype(int)

# AQ_sum
aq_cols = ["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]
pos_items = ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]
neg_items = ["BB125", "BB126", "BB127", "BB132"]
if set(aq_cols).issubset(df.columns):
    df_AQ = df[aq_cols].copy()
    # ポジティブ項目：3,4→1
    for i in pos_items:
        df_AQ[i] = df_AQ[i].replace({1: 0, 2: 0, 3: 1, 4: 1, 5: 0})
    # ネガティブ項目：1,2→1
    for i in neg_items:
        df_AQ[i] = df_AQ[i].replace({1: 1, 2: 1, 3: 0, 4: 0, 5: 0})
    df["AQ_sum"] = df_AQ.sum(axis=1)

# bullied
if set(["AB61", "AD19"]).issubset(df.columns):
    # 「本人または養育者が一方でも1回以上あった」→ bullied=1
    # （あなたのコードは <5 を1回以上と解釈しているので踏襲）
    df["bullied"] = ((df["AB61"] < 5) | (df["AD19"] < 5)).astype(int)

# 不要列の削除（存在するものだけ削る：安全運用）
drop_cols = [
    "AD57", "AD58", "AD59", "AD60", "AD61",             # 第1期PLE
    "BB39", "BB83", "OCS_sum",                          # 第2期強迫
    "AB61", "AD19",                                     # いじめ元項目
    "BB123", "BB124", "BB125", "BB126", "BB127",
    "BB128", "BB129", "BB130", "BB131", "BB132",        # AQ元項目
    "AA57", "AA58", "AA59", "AA60", "AA61",             # social cohesion元項目
    "AF37", "AF38",                                     # atopy元項目
]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(drop_cols, axis=1)
print("After feature engineering:", df.shape)


# =========================
# 4) 解析対象サブセット（あなたのコード踏襲）
# =========================
if FILTER_OCS_POSITIVE_ONLY:
    if "OCS_0or1" not in df.columns:
        raise ValueError("OCS_0or1列がありません。FILTER_OCS_POSITIVE_ONLYをFalseにするか、列名を確認してください。")
    df4 = df[df["OCS_0or1"] == 1].copy()
else:
    df4 = df.copy()

# PE（14/16歳）定義
ple_cols = ["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
            "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1"]

missing_ple = [c for c in ple_cols if c not in df4.columns]
if missing_ple:
    raise ValueError(f"PLE列が見つかりません: {missing_ple}")

ple_pos = df4[ple_cols].gt(2).any(axis=1)
ple_neg = df4[ple_cols].lt(2).all(axis=1)
df4["group"] = np.select([ple_pos, ple_neg], [1, 0], default=np.nan)

# groupが確定した人だけ
df4 = df4[df4["group"].isin([0, 1])].copy()
print("After PE definition:", df4.shape, "PE prevalence:", df4["group"].mean())


# =========================
# 5) X, y の作成
# =========================
y = df4["group"].astype(int)

drop_y = ['group']
# 予測に使わない列（あなたのコード踏襲）
drop_outcomes = [c for c in df4.columns if c.startswith("CD") or c.startswith("DD")]
drop_ocs = ["OCS_0or1"] if "OCS_0or1" in df4.columns else []

X = df4.drop(columns=drop_y + drop_outcomes + drop_ocs, errors="ignore")

# 数値化（object混入を回避）
X = X.apply(pd.to_numeric, errors="coerce")

print("X shape:", X.shape, "y shape:", y.shape)

# =========================
# 6) 欠損フラグ導入 + 欠損埋め（0）
# =========================
X_miss, miss_cols = add_missing_indicators_and_fill0(X)
print("After missing indicators:", X_miss.shape)
print("Num cols with any missing originally:", len(miss_cols))


# =========================
# 7) nested-ish CV: innerでparam、outerでAUC評価、train内でBoruta
# =========================
outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

param_grid = {
    'n_estimators': [10, 30, 50, 80, 100],
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [1, 2],
    'random_state': [RANDOM_STATE],
    'n_jobs': [int(cpu_count() / 2)]
}

scoring = make_scorer(youden_index_score, greater_is_better=True)

outer_results = []
selected_feature_lists = []

for r in range(REPEATS):
    print(f"\n===== Repeat {r+1}/{REPEATS} =====")
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_miss, y), start=1):
        X_train, X_val = X_miss.iloc[train_idx].copy(), X_miss.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        # ---- 7.1 inner CVでparams探索（train内）
        base_est = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE
        )

        grid = GridSearchCV(
            estimator=base_est,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=1
        )
        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        best_inner_score = grid.best_score_
        print(f"[Fold {fold}] best inner(Youden)={best_inner_score:.4f} params={best_params}")

        # ---- 7.2 train内でBoruta（推定器は未学習のXGB）
        boruta_est = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="auc"
        )

        selector = BorutaPy(
            estimator=boruta_est,
            n_estimators="auto",
            alpha=0.05,
            max_iter=100,
            perc=95,
            two_step=False,
            random_state=RANDOM_STATE,
            verbose=0
        )

        selector.fit(X_train.values, y_train.values)

        selected_cols = X_train.columns[selector.support_].tolist()
        tentative_cols = X_train.columns[selector.support_weak_].tolist()

        note = ""
        if len(selected_cols) == 0:
            if len(tentative_cols) > 0:
                selected_cols = tentative_cols
                note = "selected=0 so used tentative"
            else:
                selected_cols = X_train.columns.tolist()
                note = "selected=0 & tentative=0 so used all"

        # ---- 7.3 選択特徴量でfit→valでAUC
        final_model = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="auc"
        )
        final_model.fit(X_train[selected_cols], y_train)

        proba_val = final_model.predict_proba(X_val[selected_cols])[:, 1]
        auc_val = roc_auc_score(y_val, proba_val)

        print(f"[Fold {fold}] selected={len(selected_cols)} val_auc={auc_val:.3f} {note}")

        outer_results.append({
            "repeat": r + 1,
            "fold": fold,
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_features_total": int(X_train.shape[1]),
            "n_features_selected": int(len(selected_cols)),
            "inner_best_youden": float(best_inner_score),
            "val_auc": float(auc_val),
            "note": note
        })
        selected_feature_lists.append(selected_cols)

# =========================
# 8) 保存：outer結果、選択頻度、選択リスト
# =========================
res_df = pd.DataFrame(outer_results)
res_df.to_csv(OUT_OUTER, index=False)
print("\nSaved outer results:", OUT_OUTER)
print(res_df["val_auc"].describe())

# 選択リスト（行ごとに保存：列数は最大に合わせて空欄で埋める）
max_len = max(len(cols) for cols in selected_feature_lists) if selected_feature_lists else 0
lists_mat = [cols + [""] * (max_len - len(cols)) for cols in selected_feature_lists]
lists_df = pd.DataFrame(lists_mat)
lists_df.to_csv(OUT_LISTS, index=False)
print("Saved selected lists:", OUT_LISTS)

# 選択頻度
freq = Counter()
for cols in selected_feature_lists:
    freq.update(cols)

freq_df = pd.DataFrame({
    "feature": list(freq.keys()),
    "count": list(freq.values()),
    "freq": [v / len(selected_feature_lists) for v in freq.values()]
}).sort_values("count", ascending=False)

freq_df.to_csv(OUT_FREQ, index=False)
print("Saved feature frequency:", OUT_FREQ)

print("\nTop 30 features by selection frequency:")
print(freq_df.head(30))
