import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# =============================
# 設定：ファイルパス
# =============================
DATA_PATH = r"D:\documents\UT\thesis\before_impute.csv"

# 出力先（適宜変更）
OUT_DESC = r"C:\Users\sawai\PycharmProjects\ocs_ple\thesis2025\supp_PE_desc_12vars.csv"
OUT_OR   = r"C:\Users\sawai\PycharmProjects\ocs_ple\thesis2025\supp_PE_univ_logit_12vars.csv"

# =============================
# 解析対象の12変数
# =============================
vars12 = ["bullied", "AD27_7", "AA97", "AD3", "AB46", "AB250",
          "AB64", "AB54", "AA86", "AB12.5", "AB72", "AB186Ln(TD)",
          "AEIQ"]

# =============================
# 1) データ読み込み
# =============================
df0 = pd.read_csv(DATA_PATH, low_memory=False).set_index("SAMPLENUMBER")

# =============================
# 2) PE（14/16歳）作成：あなたの定義を踏襲
# =============================
ple_cols = ["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
            "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1"]

# 必要列があるかチェック
missing_cols = [c for c in ple_cols if c not in df0.columns]
if missing_cols:
    raise ValueError(f"PLE列が見つかりません: {missing_cols}")

ple_condition = df0[ple_cols].gt(2).any(axis=1)
non_condition = df0[ple_cols].lt(2).all(axis=1)

df0["PE_14_16"] = np.select([ple_condition, non_condition], [1, 0], default=np.nan)

# PEが0/1に確定した人だけ
df = df0[df0["PE_14_16"].isin([0, 1])].copy()

# ★追加：OCS_0or1==1 の人だけ
if "OCS_0or1" not in df.columns:
    raise ValueError("OCS_0or1 列が見つかりません。列名を確認してください。")
df = df[df["OCS_0or1"] == 1].copy()

# =============================
# ★追加：AB12.5 を 3群（1, 2-4, 5）に再符号化
#   - 1 を reference
#   - 2,3,4 をまとめて "2-4"
#   - 5 は "5"
# =============================
if "AB12.5" in df.columns:
    ab = pd.to_numeric(df["AB12.5"], errors="coerce")

    df["AB12.5_grp"] = np.nan
    df.loc[ab == 1, "AB12.5_grp"] = 1
    df.loc[ab.isin([2, 3, 4]), "AB12.5_grp"] = 2   # 2-4 をまとめる（内部コード2）
    df.loc[ab == 5, "AB12.5_grp"] = 5

    # 元の AB12.5 をこの新変数に置き換える（以降の処理がそのまま動くように）
    df.drop(columns=["AB12.5"], inplace=True)
    df.rename(columns={"AB12.5_grp": "AB12.5"}, inplace=True)

# =============================
# ★追加：AB64 を再符号化
#   - 5 を reference
#   - 3,4 をまとめる
# =============================
if "AB64" in df.columns:
    ab = pd.to_numeric(df["AB64"], errors="coerce")

    df["AB64_grp"] = np.nan
    df.loc[ab == 5, "AB64_grp"] = 5          # reference
    df.loc[ab.isin([3, 4]), "AB64_grp"] = 34 # 3-4 合算（内部コード34）

    # それ以外の選択肢（例：1,2）があればそのまま残す
    df.loc[ab.isin([1, 2]), "AB64_grp"] = ab[ab.isin([1, 2])]

    # 元の AB64 を置き換え
    df.drop(columns=["AB64"], inplace=True)
    df.rename(columns={"AB64_grp": "AB64"}, inplace=True)

# =============================
# ★追加：AD27_7 の 0/1 を反転
#   0 → 1
#   1 → 0
# =============================
if "AD27_7" in df.columns:
    df["AD27_7"] = pd.to_numeric(df["AD27_7"], errors="coerce")
    df.loc[df["AD27_7"].isin([0, 1]), "AD27_7"] = 1 - df.loc[df["AD27_7"].isin([0, 1]), "AD27_7"]

# =============================
# 3) bullied等があなたの前処理で作られる場合の補助
#    （すでに列があるならそのまま）
# =============================
# bulliedが無い場合のみ、あなたの定義（AB61/AD19）で作る
if "bullied" not in df.columns:
    if all(c in df.columns for c in ["AB61", "AD19"]):
        mask = df[["AB61", "AD19"]].notna().all(axis=1)
        df["bullied"] = np.nan
        df.loc[mask, "bullied"] = (df.loc[mask, ["AB61", "AD19"]].lt(5).any(axis=1)).astype(int)
    else:
        print("注意: bullied列が無く、作成に必要なAB61/AD19も見つかりませんでした。bulliedは欠損扱いになります。")
        df["bullied"] = np.nan

# 12変数の存在チェック
missing_vars = [v for v in vars12 if v not in df.columns]
if missing_vars:
    raise ValueError(f"指定された12変数のうち、データに存在しない列があります: {missing_vars}")

# =============================
# 4) 変数タイプ判定（簡易）
#    - 0/1 のみ → binary
#    - 水準が少ない整数（<=6水準） → categorical
#    - それ以外 → continuous
# =============================
def infer_type(s: pd.Series):
    x = s.dropna()
    if x.empty:
        return "unknown"
    uniq = np.sort(x.unique())
    if len(uniq) <= 2 and set(uniq).issubset({0, 1}):
        return "binary"
    # 整数っぽくて水準が少ない → カテゴリ扱い
    if pd.api.types.is_numeric_dtype(x):
        if np.all(np.isclose(x, np.round(x))) and x.nunique() <= 6:
            return "categorical"
        return "continuous"
    return "categorical"

# =============================
# 5) 記述統計（PE=0/1別）
#    - binary: n(%)
#    - continuous: mean±sd と median[IQR] 両方を出す（書き分けに便利）
#    - categorical: 水準別 n(%)（横持ちにせず“縦持ち”で出す）
# =============================
desc_rows = []

def fmt_mean_sd(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}" if len(x) >= 2 else f"{np.mean(x):.3f}"

def fmt_median_iqr(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.3f} [{q1:.3f}, {q3:.3f}]"

for v in vars12:
    vtype = infer_type(df[v])
    for g in [0, 1]:
        sub = df.loc[df["PE_14_16"] == g, v]
        n_total = sub.shape[0]
        n_nonmiss = sub.notna().sum()
        n_miss = n_total - n_nonmiss

        if vtype == "binary":
            x = sub.dropna().astype(float)
            n1 = int((x == 1).sum())
            pct1 = (n1 / len(x) * 100) if len(x) else np.nan
            value = f"{n1} ({pct1:.1f}%)" if len(x) else ""
            desc_rows.append({
                "variable": v,
                "type": vtype,
                "PE": g,
                "N_total": n_total,
                "N_nonmiss": int(n_nonmiss),
                "N_missing": int(n_miss),
                "summary": value
            })

        elif vtype == "continuous":
            x = pd.to_numeric(sub, errors="coerce").dropna()
            mean_sd = fmt_mean_sd(x) if len(x) else ""
            med_iqr = fmt_median_iqr(x) if len(x) else ""
            desc_rows.append({
                "variable": v,
                "type": vtype,
                "PE": g,
                "N_total": n_total,
                "N_nonmiss": int(n_nonmiss),
                "N_missing": int(n_miss),
                "summary_mean_sd": mean_sd,
                "summary_median_iqr": med_iqr
            })

        else:  # categorical
            x = sub.dropna()
            # 水準ごとに1行ずつ
            vc = x.value_counts(dropna=False)
            denom = vc.sum()
            for level, cnt in vc.items():
                pct = (cnt / denom * 100) if denom else np.nan
                desc_rows.append({
                    "variable": v,
                    "type": vtype,
                    "PE": g,
                    "level": level,
                    "N_total": n_total,
                    "N_nonmiss": int(n_nonmiss),
                    "N_missing": int(n_miss),
                    "summary": f"{int(cnt)} ({pct:.1f}%)"
                })

desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(OUT_DESC, index=False)
print("記述統計を保存:", OUT_DESC)

# =============================
# 6) 単変量ロジスティック回帰
#    - binary / continuous：y ~ x（連続は「1単位あたりOR」）
#      追加で「1SD増加あたりOR」も併記（比較しやすい）
#    - categorical：ダミー化して参照カテゴリに対するOR
# =============================
or_rows = []

def fit_logit(y, X):
    # yを数値化（object混入対策）
    y = pd.to_numeric(y, errors="coerce").astype(float)

    # Xを数値化（ダミー列のobject混入対策）
    X = X.copy()
    X = sm.add_constant(X, has_constant="add")

    # 重要：全列を数値(float)へ
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)

    # 念のため：欠損を落としてインデックス整合
    dat = pd.concat([y, X], axis=1).dropna()
    y2 = dat.iloc[:, 0]
    X2 = dat.iloc[:, 1:]

    model = sm.Logit(y2, X2)
    res = model.fit(disp=0)
    return res


y_all = df["PE_14_16"].astype(float)

for v in vars12:
    vtype = infer_type(df[v])

    # complete case
    dat = df[["PE_14_16", v]].copy()
    dat = dat.dropna()
    if dat.shape[0] < 30:
        # 小さすぎると推定が不安定なので注記
        or_rows.append({
            "variable": v, "type": vtype, "N_complete": dat.shape[0],
            "note": "complete caseが少ないため未推定"
        })
        continue

    y = dat["PE_14_16"].astype(float)

    if vtype in ["binary", "continuous"]:
        x = pd.to_numeric(dat[v], errors="coerce")
        dat2 = pd.DataFrame({"y": y, "x": x}).dropna()
        if dat2.shape[0] < 30:
            or_rows.append({
                "variable": v, "type": vtype, "N_complete": dat2.shape[0],
                "note": "数値変換後のcomplete caseが少ないため未推定"
            })
            continue

        # 1単位あたり
        res = fit_logit(dat2["y"], dat2[["x"]])
        beta = res.params["x"]
        se = res.bse["x"]
        p = res.pvalues["x"]
        OR = np.exp(beta)
        LCL = np.exp(beta - 1.96 * se)
        UCL = np.exp(beta + 1.96 * se)

        # 1SDあたり（連続のみ/比較用）
        sd = dat2["x"].std(ddof=1)
        if np.isfinite(sd) and sd > 0:
            OR_1sd = np.exp(beta * sd)
            LCL_1sd = np.exp((beta - 1.96 * se) * sd)
            UCL_1sd = np.exp((beta + 1.96 * se) * sd)
        else:
            OR_1sd = LCL_1sd = UCL_1sd = np.nan

        or_rows.append({
            "variable": v,
            "type": vtype,
            "N_complete": int(dat2.shape[0]),
            "OR_per_1unit": OR,
            "LCL_per_1unit": LCL,
            "UCL_per_1unit": UCL,
            "p_value": p,
            "SD_of_x": sd,
            "OR_per_1SD": OR_1sd,
            "LCL_per_1SD": LCL_1sd,
            "UCL_per_1SD": UCL_1sd
        })

    else:
        # categorical：参照カテゴリ（最頻値）をbaselineにしてダミー化
        x = dat[v].astype("category")
        # 水準を数値変換（できないものはNaN）
        levels = pd.Index(x.cat.categories)
        levels_num = pd.to_numeric(levels.astype(str), errors="coerce")

        # ★AB64 は reference=5 を強制（存在する場合）
        if v == "AB64" and (5 in pd.to_numeric(levels.astype(str), errors="coerce").values):
            ref = 5
        else:
            if np.isfinite(levels_num).any():
                ref = levels[levels_num.argmin()]
            else:
                ref = sorted(levels.astype(str))[0]

        # refを先頭に並べ替え（baselineにするため）
        x = x.cat.reorder_categories(
            [ref] + [c for c in x.cat.categories if c != ref],
            ordered=True
        )
        dummies = pd.get_dummies(x, drop_first=True)  # refが落ちてbaselineになる

        # ダミーが0列なら（実質一定）
        if dummies.shape[1] == 0:
            or_rows.append({
                "variable": v,
                "type": vtype,
                "N_complete": int(dat.shape[0]),
                "note": "カテゴリ水準が1つのみ（または一定）で未推定"
            })
            continue

        res = fit_logit(y, dummies)

        for col in dummies.columns:
            beta = res.params[col]
            se = res.bse[col]
            p = res.pvalues[col]
            OR = np.exp(beta)
            LCL = np.exp(beta - 1.96 * se)
            UCL = np.exp(beta + 1.96 * se)

            or_rows.append({
                "variable": v,
                "type": vtype,
                "N_complete": int(dat.shape[0]),
                "reference_level": ref,
                "contrast_level": col,   # 例: "2" など
                "OR": OR,
                "LCL": LCL,
                "UCL": UCL,
                "p_value": p
            })

or_df = pd.DataFrame(or_rows)
or_df.to_csv(OUT_OR, index=False)
print("単変量ロジスティック結果を保存:", OUT_OR)

# 画面にも上位だけ表示（必要なら）
print("\n--- OR結果（先頭10行） ---")
print(or_df.head(10))
