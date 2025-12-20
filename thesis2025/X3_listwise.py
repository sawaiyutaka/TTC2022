from multiprocessing import cpu_count
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# =========================
# 1) 読込
# =========================
data4grf = pd.read_table(
    "/Volumes/Pegasus32R8/TTC/2025thesis/before_impute.csv",
    delimiter=",", low_memory=False
).set_index("SAMPLENUMBER")

# =========================
# 2) あなたのdrop（同じものを踏襲）
# =========================
data4grf = data4grf.drop([
    "AB226", "AB227", "AB228", "AB229",  # 第二次性徴
    "AA185", "AA186", "AA187", "AA188",  # 両親アルコール
    "AD36", "AD37", "AD38", "AD39", "AD40", "AD41", "AD42", "AD43", "AD44", "AD45", "AD46", "AD47", "AD48",
    "AD49", "AD50", "AD51", "AD52", "AD53", "AD54", "AD55", "AD56",  # CPAQ
    "AC28", "AC28", "AC29", "AC30", "AC31", "AC32",  # 子WHO5
    "AA205", "AA206", "AA207", "AA208", "AA209", "AA210",  # 母K6
    "AB202", "AB203", "AB204", "AB205", "AB206", "AB207", "AB208",
    "AB209", "AB210", "AB211", "AB212", "AB213", "AB214", "AB215",  # 母PPL
    "AA213A", "AA213NIN", "AA213B", "AA214A", "AA214NIN", "AA214B", "AA215A", "AA215NIN",
    "AA215B", "AA216A", "AA216NIN", "AA216B", "AA217A", "AA217NIN", "AA217B", "AA218A", "AA218NIN", "AA218B",  # 母SSQ
    "AB35", "AB36", "AB37", "AB38", "AB39", "AB40", "AB41", "AB42", "AB43", "AB44",  # webaddiction
    "VS9", "VS10", "VS11", "VS12",  # SelfRegulation
    "AE6", "AE7", "AE8", "AE9",  # 2D4D
    "AC81", "AC82", "AC83", "AC84", "AC85", "AC86", "AC87", "AC88", "AC89",  # 子time discount
    "AB186", "AB187", "AB188", "AB189", "AB190", "AB191", "AB192", "AB193", "AB194"  # 母time discount
], axis=1)

# =========================
# 3) アウトカム（PE判定に使うPLE列群）を準備
# =========================
outcome = data4grf.filter(regex='^(C|D)', axis=1)

# =========================
# 4) 欠損率<5%の列だけ抽出（あなたの方針踏襲）
# =========================
col = len(data4grf.columns)
NUM_0F_NAN = int(col * 0.05)

sr = data4grf.isnull().sum()
sr = sr[sr < NUM_0F_NAN]  # 欠損が閾値未満の列だけ残す
df1 = data4grf[sr.index].copy()

# =========================
# 5) 欠損補完前の結合＋特徴量作成（あなたの元コード踏襲）
# =========================
df_demo = df1.join(outcome, how="left")  # ここはdf1にoutcome列が含まれない前提ならOK

# -- 特徴量作成（同じ） --
social_cols = [f"AA{num}" for num in range(57, 62)]
if set(social_cols).issubset(df_demo.columns):
    df_demo["social_cohesion"] = df_demo[social_cols].sum(axis=1)

if set(["AF37", "AF38"]).issubset(df_demo.columns):
    df_demo["atopy"] = ((df_demo[["AF37", "AF38"]] == 1).any(axis=1)).astype(int)

pos_items = ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]
neg_items = ["BB125", "BB126", "BB127", "BB132"]
if set(pos_items + neg_items).issubset(df_demo.columns):
    df_demo[pos_items] = df_demo[pos_items].applymap(lambda x: 1 if x in (3, 4) else 0)
    df_demo[neg_items] = df_demo[neg_items].applymap(lambda x: 1 if x in (1, 2) else 0)
    df_demo["AQ_sum"] = df_demo[pos_items + neg_items].sum(axis=1)

if set(["AB61", "AD19"]).issubset(df_demo.columns):
    mask = df_demo[["AB61", "AD19"]].notna().all(axis=1)
    df_demo["bullied"] = np.nan
    df_demo.loc[mask, "bullied"] = (
        df_demo.loc[mask, ["AB61", "AD19"]].lt(5).any(axis=1)
    ).astype(int)

# 不要列削除（存在するものだけ落とす：listwise版では安全運用）
drop_initial = (
    social_cols
    + pos_items
    + neg_items
    + ["AF37", "AF38", "AB61", "AD19"]
    + ["AD57", "AD58", "AD59", "AD60", "AD61"]  # 第1期PLE用
    + ["BB39", "BB83", "OCS_sum"]               # 第2期強迫
)
drop_initial = [c for c in drop_initial if c in df_demo.columns]
df_demo.drop(columns=drop_initial, inplace=True)

# =========================
# 6) PE（group）作成（あなたの定義踏襲）
# =========================
ple_cols = [
    "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
    "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1",
]

missing_ple = [c for c in ple_cols if c not in df_demo.columns]
if missing_ple:
    raise ValueError(f"PLE列が見つかりません: {missing_ple}")

ple_condition = df_demo[ple_cols].gt(2).any(axis=1)
non_condition = df_demo[ple_cols].lt(2).all(axis=1)

df_demo["group"] = np.select([ple_condition, non_condition], [1, 0], default=np.nan)

# groupが0/1の人だけに限定
df_use = df_demo[df_demo["group"].isin([0, 1])].copy()

y = df_use["group"].astype(int)
X = df_use.drop(columns=["group"] + ple_cols)

# =========================
# 7) ★ここがlistwise deletion：Xの欠損が1つでもある行を落とす
# =========================
before_n = X.shape[0]
complete_mask = X.notna().all(axis=1)
X_cc = X.loc[complete_mask].copy()
y_cc = y.loc[complete_mask].copy()
after_n = X_cc.shape[0]

print(f"[Listwise deletion] N: {before_n} → {after_n} （除外 {before_n - after_n}）")

# 念のため：欠損が残っていないことを確認
assert X_cc.isna().sum().sum() == 0

# =========================
# 8) XGBoost（ひとまず妥当な設定。元論文のbest paramsがあればここに入れ替え）
# =========================
xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1,
    gamma=0,
    objective="binary:logistic",
    eval_metric="auc",
    n_jobs=int(cpu_count() * 4 / 5),
    random_state=42
)

# =========================
# 9) CVで予測確率→AUC（感度分析に必要な最低限）
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

proba = cross_val_predict(
    xgb, X_cc, y_cc,
    cv=cv,
    method="predict_proba"
)[:, 1]

auc = roc_auc_score(y_cc, proba)
print(f"[Listwise deletion] CV AUC = {auc:.3f}")

# =========================
# 10) Youden指数でカットオフ（あなたの運用に合わせるなら）
# =========================
fpr, tpr, thr = roc_curve(y_cc, proba)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_thr = thr[best_idx]
print(f"Youden best threshold = {best_thr:.4f}")

pred = (proba >= best_thr).astype(int)
cm = confusion_matrix(y_cc, pred)
tn, fp, fn, tp = cm.ravel()
sens = tp / (tp + fn) if (tp + fn) else np.nan
spec = tn / (tn + fp) if (tn + fp) else np.nan
print("Confusion matrix:\n", cm)
print(f"Sensitivity={sens:.3f}, Specificity={spec:.3f}")

# =========================
# 11) 出力（Supplementary用に）
# =========================
pd.DataFrame({
    "SAMPLENUMBER": X_cc.index,
    "PE": y_cc.values,
    "proba": proba
}).to_csv("/Volumes/Pegasus32R8/TTC/2025thesis/sens_listwise_predictions.csv", index=False)

pd.DataFrame({
    "metric": ["AUC", "Youden_threshold", "Sensitivity", "Specificity", "N_complete_case"],
    "value": [auc, best_thr, sens, spec, after_n]
}).to_csv("/Volumes/Pegasus32R8/TTC/2025thesis/sens_listwise_summary.csv", index=False)

print("Saved: sens_listwise_predictions.csv, sens_listwise_summary.csv")


