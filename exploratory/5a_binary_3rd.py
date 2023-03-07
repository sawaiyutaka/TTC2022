import pandas as pd
from missingpy import MissForest
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


df0 = pd.read_table("/Volumes/Pegasus32R8/TTC/2022csv_boruta/imputed.csv", delimiter=",")
df0 = df0.set_index("SAMPLENUMBER")
print(df0)

df0 = df0[df0['OCS_0or1'] == 1]
print(df0)

# 第２期のAQ素点からAQを計算
# AQの合計点を作成
df_AQ = df0[["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"]].copy()
print(df_AQ)

for i in ["BB123", "BB124", "BB128", "BB129", "BB130", "BB131"]:
    df_AQ = df_AQ.replace({i: {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}})
    print(df_AQ)

for i in ["BB125", "BB126", "BB127", "BB132"]:
    df_AQ = df_AQ.replace({i: {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}})
    print(df_AQ)

df0["AQ_sum"] = df_AQ.sum(axis=1)
print("第2回AQ合計\n", df0["AQ_sum"])

# PLEで全項目回答なしは除外
df0 = df0.dropna(subset=["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"],
                 how='all')
print("PLEで全項目回答なしは除外\n", df0)

# 第4期のPLEを除外
df = df0.drop(["DD64_1", "DD65_1", "DD66_1", "DD67_1", "DD68_1", "DD69_1", "DD70_1", "DD71_1", "DD72_1"], axis=1)

# 第２期のAQ素点を除外
df = df.drop(["BB123", "BB124", "BB125", "BB126", "BB127", "BB128", "BB129", "BB130", "BB131", "BB132"], axis=1)

# 1つでも「1回以上あった」があった人をPLEありとする
df_ple_3rd = df[
    (df['CD57_1'] > 1) | (df['CD58_1'] > 1) | (df['CD59_1'] > 1) | (df['CD60_1'] > 1) | (df['CD61_1'] > 1)
    | (df['CD62_1'] > 1) | (df['CD63_1'] > 1) | (df['CD64_1'] > 1) | (df['CD65_1'] > 1)].copy()
print(df_ple_3rd)
df_ple_3rd["group_3rd"] = 1

# 全ての項目に回答があって、「なかった」のみの人はPLEなしとする(300427は除外される)
df_non_3rd = df[
    (df['CD57_1'] == 1) & (df['CD58_1'] == 1) & (df['CD59_1'] == 1) & (df['CD60_1'] == 1) & (df['CD61_1'] == 1)
    & (df['CD62_1'] == 1) & (df['CD63_1'] == 1) & (df['CD64_1'] == 1) & (df['CD65_1'] == 1)].copy()
print(df_non_3rd)
df_non_3rd["group_3rd"] = 0

df_3rd = pd.concat([df_ple_3rd, df_non_3rd])
print(df_3rd)

# 第3期のPLE素点を除外
df_3rd = df_3rd.drop(
    ["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"],
    axis=1)

# 第2期のocsを除外
df_3rd = df_3rd.drop(
    ["OCS_sum", "OCS_0or1", "BB39", "BB56", "BB57", "BB73", "BB83", "BB95", "BB96", "BB116"], axis=1)

"""
# PLEの合計点を作成(第3期)
df_Y = df_3rd[["CD57_1", "CD58_1", "CD59_1", "CD60_1", "CD61_1", "CD62_1", "CD63_1", "CD64_1", "CD65_1"]].copy()
print("df_Y\n", df_Y)
df_3rd["PLE_sum_3rd"] = df_Y.sum(axis=1)
print("第3回PLE合計\n", df_3rd["PLE_sum_3rd"])

print(df_3rd)
"""

"""
# 第1期の強迫、PLEを除外
df_3rd = df_3rd.drop(["AB71", "AB87", "AB88", "AB104", "AB114", "AB126", "AB127", "AB145"], axis=1)
df_3rd = df_3rd.drop(["AD57", "AD58", "AD59", "AD60", "AD61", "AD62"], axis=1)
"""

# df_3rd = df_3rd.loc[:, ~df_3rd.columns.duplicated()]
# print("重複を削除\n", df_3rd)
print(df_3rd)
df_3rd.to_csv("/Volumes/Pegasus32R8/TTC/2022csv_boruta/binary_3rd.csv")
