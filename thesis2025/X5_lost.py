"""
目的：
- OCSあり群（OCS_0or1==1）について
  ① 14/16歳PEが判定できて主解析に入った群（included：group=0/1）
  ② PEが判定できず主解析から外れた群（excluded：group=999）
  のベースライン特性（性別・SES・症状など）を比較する表を作る
- さらに「既知のPEイベント数」を可能な範囲で集計して保存する

重要な設計：
- 共変量（X）は欠損<5%で絞る（あなたの方針）
- ただし PLE列（CD57_1など）は欠損率で落とさない（group判定に必須）
- join時の列名重複（OCS_0or1など）を自動で回避する

出力：
- supp_attrition_table_ocs_positive.csv
- supp_attrition_known_PE_counts.csv
"""

import pandas as pd
import numpy as np
import sys
import sklearn.neighbors._base

# sklearnのバージョン差対処（missingpy等の依存で必要だった名残として）
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

# =========================
# 0) パス
# =========================
IN_PATH = "/Volumes/Pegasus32R8/TTC/2025thesis/before_impute.csv"
OUT_MAIN = "/Volumes/Pegasus32R8/TTC/2025thesis/supp_attrition_table_ocs_positive.csv"
OUT_PEEK = "/Volumes/Pegasus32R8/TTC/2025thesis/supp_attrition_known_PE_counts.csv"


# =========================
# 1) データ読み込み
# =========================
data4grf = pd.read_table(IN_PATH, delimiter=",", low_memory=False).set_index("SAMPLENUMBER")
print("Loaded:", data4grf.shape)

# =========================
# 2) 不要列drop（あなたの元コード踏襲）
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
], axis=1, errors="ignore")
print("After drop:", data4grf.shape)

# =========================
# 3) PLE列（欠損率で落とさない）
# =========================
ple_cols = [
    "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
    "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1",
]
missing_ple = [c for c in ple_cols if c not in data4grf.columns]
if missing_ple:
    raise ValueError(f"元データにPLE列が見つかりません: {missing_ple}")

ple_df = data4grf[ple_cols].copy()

# =========================
# 4) OCS列（OCSあり群抽出に必須）
# =========================
if "OCS_0or1" not in data4grf.columns:
    cand = [c for c in data4grf.columns if "OCS" in str(c).upper()]
    raise ValueError(f"OCS_0or1列が見つかりません。候補: {cand}")

# =========================
# 5) 共変量：欠損<5%の列だけ残す（あなたの方針）
#    ※ただしple_colsは別で後から必ず足す
# =========================
col_count = len(data4grf.columns)
NUM_0F_NAN = int(col_count * 0.05)

sr = data4grf.isnull().sum()
sr = sr[sr < NUM_0F_NAN]
df1 = data4grf[sr.index].copy()
print("Covariates under missing cutoff:", df1.shape)

# =========================
# 6) df_demo構築：df1に ple_df を追加（列重複は自動回避）
# =========================
to_join = ple_df.loc[:, ~ple_df.columns.isin(df1.columns)]
df_demo = df1.join(to_join, how="left")

# OCS_0or1 は df1 に無いときだけ足す（重複回避）
if "OCS_0or1" not in df_demo.columns:
    df_demo = df_demo.join(data4grf[["OCS_0or1"]], how="left")

print("df_demo:", df_demo.shape)

# =========================
# 7) 特徴量作成（あなたの元コード踏襲）
# =========================
# social cohesion
social_cols = [f"AA{num}" for num in range(57, 62)]
if set(social_cols).issubset(df_demo.columns):
    df_demo["social_cohesion"] = df_demo[social_cols].sum(axis=1)

# atopy
if set(["AF37", "AF38"]).issubset(df_demo.columns):
    df_demo["atopy"] = ((df_demo[["AF37", "AF38"]] == 1).any(axis=1)).astype(int)

# AQ_sum
pos_items = ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]
neg_items = ["BB125", "BB126", "BB127", "BB132"]
if set(pos_items + neg_items).issubset(df_demo.columns):
    df_demo[pos_items] = df_demo[pos_items].applymap(lambda x: 1 if x in (3, 4) else 0)
    df_demo[neg_items] = df_demo[neg_items].applymap(lambda x: 1 if x in (1, 2) else 0)
    df_demo["AQ_sum"] = df_demo[pos_items + neg_items].sum(axis=1)

# bullied
if set(["AB61", "AD19"]).issubset(df_demo.columns):
    mask = df_demo[["AB61", "AD19"]].notna().all(axis=1)
    df_demo["bullied"] = np.nan
    df_demo.loc[mask, "bullied"] = (df_demo.loc[mask, ["AB61", "AD19"]].lt(5).any(axis=1)).astype(int)

# 元項目削除（存在するものだけ安全に）
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
# 8) group（PE 0/1/999）を作成
# =========================
ple_condition = df_demo[ple_cols].gt(2).any(axis=1)
non_condition = df_demo[ple_cols].lt(2).all(axis=1)

df_demo["group"] = np.select([ple_condition, non_condition], [1, 0], default=999)

# =========================
# 9) 「既知PE」ラベル（可能な範囲）
# =========================
ple_vals = df_demo[ple_cols]
any_pos_observed = ple_vals.gt(2).any(axis=1)
any_observed = ple_vals.notna().any(axis=1)
all_observed_nonpos = ple_vals.lt(2).where(ple_vals.notna(), True).all(axis=1)

df_demo["known_PE"] = np.nan
df_demo.loc[any_pos_observed, "known_PE"] = 1
df_demo.loc[(~any_pos_observed) & any_observed & all_observed_nonpos, "known_PE"] = 0

# =========================
# 10) OCSあり群で included/excluded を作る
# =========================
df_ocs = df_demo[df_demo["OCS_0or1"] == 1].copy()
df_ocs["analysis_included"] = np.where(df_ocs["group"].isin([0, 1]), 1, 0)

print("OCSあり総数:", df_ocs.shape[0])
print("included:", int(df_ocs["analysis_included"].sum()),
      "excluded:", int((df_ocs["analysis_included"] == 0).sum()))

# =========================
# 11) 表に入れるベースライン変数（あなたのデータに合わせて追加）
#    ※存在する列だけ採用
# =========================
BASELINE_VARS_CANDIDATES = [
    # まずあなたの作成変数（入れやすい）
    "bullied", "AQ_sum", "social_cohesion", "atopy",
    # OCS重症度があるなら優先（列名が違う場合はここに追加）
    "OCS_sum",
    # 例：性別・SES（列名をあなたの実データに合わせて追加）
    "SEX", "sex", "Gender", "gender", "AB195", "income", "father_edu", "mother_edu",
]
baseline_vars = [v for v in BASELINE_VARS_CANDIDATES if v in df_ocs.columns]
print("使用するベースライン変数:", baseline_vars)

# 探索用（性別/収入/学歴の候補列を出す）
print("\n[探索] sex/edu/inc を含む列候補（先頭30）")
for key in ["sex", "gender", "edu", "educ", "inc", "income"]:
    hits = [c for c in df_ocs.columns if key in str(c).lower()]
    if hits:
        print(key, ":", hits[:30])

# =========================
# 12) 記述統計（included vs excluded）
# =========================
def infer_type(s: pd.Series):
    x = s.dropna()
    if x.empty:
        return "unknown"
    uniq = np.sort(x.unique())
    if len(uniq) <= 2 and set(uniq).issubset({0, 1}):
        return "binary"
    if pd.api.types.is_numeric_dtype(x):
        return "continuous"
    return "categorical"

def mean_sd(x):
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) >= 2 else (f"{x.mean():.3f}" if len(x) else "")

def median_iqr(x):
    if len(x) == 0:
        return ""
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.3f} [{q1:.3f}, {q3:.3f}]"

rows = []
for v in baseline_vars:
    vtype = infer_type(df_ocs[v])
    for g in [1, 0]:
        sub = df_ocs.loc[df_ocs["analysis_included"] == g, v]
        n_total = sub.shape[0]
        n_nonmiss = int(sub.notna().sum())
        n_miss = int(n_total - n_nonmiss)

        if vtype == "binary":
            x = pd.to_numeric(sub, errors="coerce").dropna()
            n1 = int((x == 1).sum())
            pct = (n1 / len(x) * 100) if len(x) else np.nan
            summ = f"{n1} ({pct:.1f}%)" if len(x) else ""
            rows.append({
                "variable": v, "type": vtype,
                "group": "included" if g == 1 else "excluded",
                "N_total": n_total, "N_nonmiss": n_nonmiss, "N_missing": n_miss,
                "summary": summ
            })
        else:
            x = pd.to_numeric(sub, errors="coerce").dropna()
            rows.append({
                "variable": v, "type": "continuous",
                "group": "included" if g == 1 else "excluded",
                "N_total": n_total, "N_nonmiss": n_nonmiss, "N_missing": n_miss,
                "mean_sd": mean_sd(x),
                "median_iqr": median_iqr(x)
            })

table_df = pd.DataFrame(rows)
table_df.to_csv(OUT_MAIN, index=False)
print("\nSaved baseline comparison table:", OUT_MAIN)

# =========================
# 13) 既知PEイベント数（included/excluded別）
# =========================
pe_counts = (
    df_ocs.groupby(["analysis_included", "known_PE"], dropna=False)
    .size()
    .reset_index(name="n")
)
pe_counts["analysis_included"] = pe_counts["analysis_included"].map({1: "included", 0: "excluded"})
pe_counts["known_PE_label"] = pe_counts["known_PE"].map({1.0: "PE_known_positive", 0.0: "PE_known_negative"}).fillna("PE_unknown")

pe_counts.to_csv(OUT_PEEK, index=False)
print("Saved known-PE counts:", OUT_PEEK)
print(pe_counts)

import pandas as pd
import numpy as np

# =========================
# 1) OCSあり群に限定
# =========================
df_ocs = df_demo[df_demo["OCS_0or1"] == 1].copy()

# group をラベル化
df_ocs["PE_group"] = df_ocs["group"].map({
    1: "PE+",
    0: "PE-",
    999: "PE_unknown"
})

print(df_ocs["PE_group"].value_counts(dropna=False))

# =========================
# 2) 今回使う変数
# =========================
vars_target = ["bullied", "AB195", "AQ_sum", "AEIQ", "TTC_sex"]

vars_exist = [v for v in vars_target if v in df_ocs.columns]
print("解析に使う変数:", vars_exist)

# =========================
# 3) 記述統計用の関数
# =========================
def summarize_binary(x):
    x = x.dropna()
    if len(x) == 0:
        return ""
    n1 = (x == 1).sum()
    return f"{n1} ({n1/len(x)*100:.1f}%)"

def summarize_continuous(x):
    x = x.dropna()
    if len(x) == 0:
        return {"mean_sd": "", "median_iqr": ""}
    mean_sd = f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    median_iqr = f"{med:.2f} [{q1:.2f}, {q3:.2f}]"
    return {"mean_sd": mean_sd, "median_iqr": median_iqr}

# =========================
# 4) 表の作成
# =========================
rows = []

for v in vars_exist:
    for g in ["PE+", "PE-", "PE_unknown"]:
        sub = df_ocs.loc[df_ocs["PE_group"] == g, v]
        n_total = sub.shape[0]
        n_nonmiss = sub.notna().sum()

        # binaryかcontinuousかを簡易判定
        uniq = sub.dropna().unique()
        if set(uniq).issubset({0, 1}):
            summary = summarize_binary(sub)
            rows.append({
                "variable": v,
                "group": g,
                "N_total": n_total,
                "N_nonmiss": n_nonmiss,
                "summary": summary
            })
        elif set(uniq).issubset({1, 2}):
            summary = summarize_binary(sub)
            rows.append({
                "variable": v,
                "group": g,
                "N_total": n_total,
                "N_nonmiss": n_nonmiss,
                "summary": summary
            })
        else:
            s = summarize_continuous(sub)
            rows.append({
                "variable": v,
                "group": g,
                "N_total": n_total,
                "N_nonmiss": n_nonmiss,
                "mean_sd": s["mean_sd"],
                "median_iqr": s["median_iqr"]
            })

table_attrition = pd.DataFrame(rows)

# =========================
# 5) 保存
# =========================
out_path = "/Volumes/Pegasus32R8/TTC/2025thesis/supp_baseline_ocs_PE3groups.csv"
table_attrition.to_csv(out_path, index=False)

print("Saved:", out_path)


