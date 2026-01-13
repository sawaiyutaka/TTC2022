import pandas as pd
import numpy as np


def calc_or_woolf(a, b, c, d):
    # セル0がある場合の0.5補正（Haldane-Anscombe）
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lcl = np.exp(np.log(or_) - 1.96 * se)
    ucl = np.exp(np.log(or_) + 1.96 * se)
    return or_, lcl, ucl


# =========================================================
# 1) 読込
# =========================================================
data4grf = pd.read_table(
    r"D:\documents\UT\thesis\before_impute.csv",
    delimiter=",", low_memory=False
).set_index("SAMPLENUMBER")

# （必要なら）あなたのdrop処理をここで実行
# data4grf = data4grf.drop([...], axis=1)

# =========================================================
# 2) 14/16歳PE（PLE）定義：あなたのロジックを流用
# =========================================================
ple_cols = [
    "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
    "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1",
]

# 必要列だけ抜き出して作業（ここが大事：余計な列を混ぜない）
df = data4grf[ple_cols].copy()

ple_condition = df[ple_cols].gt(2).any(axis=1)
non_condition = df[ple_cols].lt(2).all(axis=1)

# 解析用PE（二値）。判定不能は NaN にして落とす
df["PE_14_16"] = np.select([ple_condition, non_condition], [1, 0], default=np.nan)

# =========================================================
# 3) 12歳OCS（二値）を足す（列名はあなたのデータに合わせて）
# =========================================================
# まず候補を表示して、正しい列名に差し替え
ocs_candidates = [c for c in data4grf.columns if "OCS" in str(c).upper()]
print("OCS候補列:", ocs_candidates)

OCS_12_COL = "OCS_0or1"  # ←ここだけあなたの実データの列名に変更
df["OCS_12_bin"] = data4grf[OCS_12_COL]

# =========================================================
# 4) 欠損除外して2×2分割表
# =========================================================
df_or = df[["OCS_12_bin", "PE_14_16"]].dropna().copy()

# 0/1になっているか確認（念のため）
print("OCS分布:\n", df_or["OCS_12_bin"].value_counts(dropna=False))
print("PE分布:\n", df_or["PE_14_16"].value_counts(dropna=False))

tab = pd.crosstab(df_or["OCS_12_bin"], df_or["PE_14_16"])
tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
tab.index = ["OCS=0", "OCS=1"]
tab.columns = ["PE=0", "PE=1"]
print("\n=== 2×2分割表（観測値ベース）===\n", tab)

# a,b,c,d
a = tab.loc["OCS=1", "PE=1"]
b = tab.loc["OCS=1", "PE=0"]
c = tab.loc["OCS=0", "PE=1"]
d = tab.loc["OCS=0", "PE=0"]

or_, lcl, ucl = calc_or_woolf(a, b, c, d)
print(f"\n粗OR = {or_:.3f} (95%CI {lcl:.3f} - {ucl:.3f})")

# 保存（supp用）
# tab.to_csv("/Volumes/Pegasus32R8/TTC/2025thesis/OCS12_vs_PE1416_2x2.csv")
# df_or.to_csv("/Volumes/Pegasus32R8/TTC/2025thesis/OCS12_vs_PE1416_analysis_dataset.csv")

import statsmodels.api as sm

df_lr = df_or.copy()
X = sm.add_constant(df_lr["OCS_12_bin"].astype(float))
y = df_lr["PE_14_16"].astype(float)

model = sm.Logit(y, X).fit(disp=0)
coef = model.params["OCS_12_bin"]
se = model.bse["OCS_12_bin"]
or_lr = np.exp(coef)
lcl_lr = np.exp(coef - 1.96 * se)
ucl_lr = np.exp(coef + 1.96 * se)

print(f"Logit OR = {or_lr:.3f} (95%CI {lcl_lr:.3f} - {ucl_lr:.3f})")
print(model.summary())
