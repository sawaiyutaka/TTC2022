from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import seaborn as s
from matplotlib import pyplot as plt
from missingpy import MissForest
import sys
import sklearn.neighbors._base
# sklearnのバージョンによって、.baseが_.base となったことによるエラーに対処
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

data4grf = pd.read_table("/Volumes/Pegasus32R8/TTC/2025thesis/before_impute.csv",
                         delimiter=",", low_memory=False)
data4grf = data4grf.set_index("SAMPLENUMBER")
print(data4grf)

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

col = len(data4grf)
NUM_0F_NAN = int(col * 0.05)  # 欠損値が何人未満の項目を使うか

print("各列のNaN個数\n", data4grf.isnull().sum())

# 第3期と第4期のPLEは欠損値の扱いが異なるため、分けておく
outcome = data4grf.filter(regex='^(C|D)', axis=1)
print(outcome)

# 欠損値がNUM OF NAN未満の項目のみ抽出
sr = data4grf.isnull().sum()
sr.to_csv("/Volumes/Pegasus32R8/TTC/2025thesis/NaN_in_data.csv")

s.set()
s.displot(sr)
sr = sr[sr < NUM_0F_NAN]

print(sr)
print(sr.index)  # 共変量の中でNaNが●個未満の項目名を抽出
name_columns = pd.DataFrame(sr, columns=["num_of_NaN"])
print("共変量の中でNaNが規定未満の項目\n", name_columns)

df1 = data4grf[sr.index]
print("columns under cutoff\n", df1)

# 参加者ごとの欠損値を表示
print("NaN個数\n", df1.isnull().sum(axis=1))
sr2 = df1.isnull().sum(axis=1) / len(df1.columns) * 100
s.set()
s.displot(sr2)
# plt.show()


# 欠損値補完前のTable1作成
# ─── データ結合 ─────────────────────────────────────────
# df1：補完前の説明変数データフレーム
# outcome：アウトカム（OCS_0or1など）を含むデータフレーム
df_demo = df1.join(outcome)

# ─── 特徴量作成 ─────────────────────────────────────────
# 1. 社会的結束（social cohesion）の合計
social_cols = [f"AA{num}" for num in range(57, 62)]
df_demo["social_cohesion"] = df_demo[social_cols].sum(axis=1)

# 2. アトピー（atopy）の二値化
df_demo["atopy"] = (
    (df_demo[["AF37", "AF38"]] == 1).any(axis=1)
).astype(int)

# 3. AQ合計点の算出
pos_items = ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]
neg_items = ["BB125", "BB126", "BB127", "BB132"]

# ポジティブ項目は3,4を1に、それ以外を0に
df_demo[pos_items] = df_demo[pos_items].applymap(lambda x: 1 if x in (3, 4) else 0)
# ネガティブ項目は1,2を1に、それ以外を0に
df_demo[neg_items] = df_demo[neg_items].applymap(lambda x: 1 if x in (1, 2) else 0)

df_demo["AQ_sum"] = df_demo[pos_items + neg_items].sum(axis=1)

# 4. いじめ経験（bullied）の定義
mask = df_demo[["AB61", "AD19"]].notna().all(axis=1)
df_demo["bullied"] = np.nan
df_demo.loc[mask, "bullied"] = (
    df_demo.loc[mask, ["AB61", "AD19"]].lt(5).any(axis=1)
).astype(int)

# ─── 不要列の一括削除 ─────────────────────────────────────────
drop_initial = (
    social_cols
    + pos_items
    + neg_items
    + ["AF37", "AF38", "AB61", "AD19"]
    + ["AD57", "AD58", "AD59", "AD60", "AD61"]  # 第1期PLE用
    + ["BB39", "BB83", "OCS_sum"]               # 第2期強迫
)
df_demo.drop(columns=drop_initial, inplace=True)

# ─── PLEあり／なしでサブセット ─────────────────────────────────────────
ple_cols = [
    "CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1",
    "DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1",
]

# 各行の条件を Boolean Series で作成
ple_condition = df_demo[ple_cols].gt(2).any(axis=1)
non_condition = df_demo[ple_cols].lt(2).all(axis=1)

# np.select で一度に group 列を作成（その他は 999）
df_demo["group"] = np.select(
    [ple_condition, non_condition],
    [1, 0],
    default=999
)

df_concat = df_demo.copy()  # すべてのケースを含む場合
# — もし group が 1 or 0 の人だけ使いたいなら：
# df_concat = df_demo[df_demo["group"].isin([0,1])].copy()

# ─── X, y の抽出と結合 ─────────────────────────────────────────
y = df_concat["group"]
print("y")
print(y)
X = df_concat.drop(columns=["group"] + ple_cols)
print("X")
print(X)
# 630項目＋使わなかったPLE4項目x2（CD62_1	CD63_1	CD64_1	CD65_1	DD69_1	DD70_1	DD71_1	DD72_1）

# 欠損値補完前の X と y を横方向に結合して保存
df_Xy = pd.concat([X, y], axis=1)
df_Xy.to_csv("/Volumes/Pegasus32R8/TTC/2025thesis/Xy_before_imputation.csv", index=False)

"""
# Make an instance and perform the imputation
imputer = MissForest(criterion='squared_error', max_features=1.0, n_jobs=int(cpu_count() * 4 / 5))

# https://pypi.org/project/missingpy/
# 説明変数だけmissforestで補完
df_imputed = imputer.fit_transform(df1)
print("df_imputed\n", df_imputed)

# 各列を整数に丸める（身長、体重も丸め）
df_imputed = df_imputed.round().astype(int)

# 第3, 第4期のPLE（欠損値あり）と統合
df1[df1.columns.values] = df_imputed

print("after missforest", df1)

# アウトカムと結合
df2 = pd.merge(df1, outcome, left_index=True, right_index=True)
print("1期、2期のデータの内、NaNが5%未満の項目のみ抽出した\n", df2)
df2.to_csv("/Volumes/Pegasus32R8/TTC/2025thesis/X_NAN_under_5percent_and_Y.csv")
"""